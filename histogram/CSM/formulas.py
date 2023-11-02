import numpy as np
import pandas as pd
import math

def read_pdb(file):
    """
    Extracting data from pdb file as DataFrame
    
    Args:
        file (str): name of the pdb file to extract data from.
        
    Returns:
        df (DataFrame): DataFrame with data from pdb file.
    """
            
    head = ['MODEL', 'ATOM NO', 'ATOM ID', 'Confmn', 'RESIDUE', 'RES SEQ','RES INSERT','X AXIS', 'Y AXIS', 'Z AXIS', 'NR', 'Atm']
    numb = ['MODEL', 'ATOM ID', 'Confmn', 'RESIDUE', 'RES SEQ', 'NR', 'Atm']
    spaces = [(0, 6), (6, 11), (12, 16), (16, 17), (17, 20), (21, 22), (22, 26), (31, 38), (38, 46), (46, 54), (54, 78), (76,78)]

    df = pd.read_fwf(file, colspecs=spaces, names=head)
    df = df[df['MODEL'] == 'ATOM'].drop(['MODEL', 'NR'], 1)
    for j in set(df.columns) - set(numb): df[j] = pd.to_numeric(df[j])
    df = df.replace(np.nan,0)
    df = df[df['Confmn'] == 0]
    return df

def read_pdb_water(file):
    """
    Extracting water molecules data from pdb file as DataFrame
    
    Args:
        file (str): name of the pdb file to extract data from.
        
    Returns:
        OH (DataFrame): DataFrame with water molecules data.
    """
    
    head = ['MODEL', 'ATOM NO', 'ATOM ID', 'Confmn', 'RESIDUE', 'RES SEQ','RES INSERT','X AXIS', 'Y AXIS', 'Z AXIS', 'NR', 'Atm']

    numb = ['MODEL', 'ATOM ID', 'Confmn', 'RESIDUE', 'RES SEQ', 'NR', 'Atm']

    spaces = [(0, 6), (6, 11), (12, 16), (16, 17), (17, 20), (21, 22), (22, 26), (31, 38), (38, 46), (46, 54), (54, 78), (76,78)]
    
    df = pd.read_fwf(file, colspecs=spaces, names=head)
    df = df[df['MODEL'] == 'HETATM'].drop(['NR', 'MODEL'], 1)
    for j in set(df.columns) - set(numb): df[j] = pd.to_numeric(df[j])
    df = df.replace(np.nan,0)
    OH = df.loc[ (df['Atm'] == 'O') & (df['RESIDUE'] == 'HOH') & (df['Confmn'] == 0)]
    return OH

def read_pdb_metals(file):
    """
    Extracting water molecules data from pdb file as DataFrame
    
    Args:
        file (str): name of the pdb file to extract data from.
        
    Returns:
        OH (DataFrame): DataFrame with water molecules data.
    """
    
    head = ['MODEL', 'ATOM NO', 'ATOM ID', 'Confmn', 'RESIDUE', 'RES SEQ','RES INSERT','X AXIS', 'Y AXIS', 'Z AXIS', 'NR', 'Atm']

    numb = ['MODEL', 'ATOM ID', 'Confmn', 'RESIDUE', 'RES SEQ', 'NR', 'Atm']

    spaces = [(0, 6), (6, 11), (12, 16), (16, 17), (17, 20), (21, 22), (22, 26), (31, 38), (38, 46), (46, 54), (54, 78), (76,78)]
    
    df = pd.read_fwf(file, colspecs=spaces, names=head)
    df = df[df['MODEL'] == 'HETATM'].drop(['NR', 'MODEL'], 1)
    for j in set(df.columns) - set(numb): df[j] = pd.to_numeric(df[j])
    df = df.replace(np.nan,0)
    metals = ['FE', 'ZN', 'AU', 'CU', 'HG', 'AS', 'CD', 'CO', 'NI', 'PT',
              'SE', 'MO', 'MN', 'K', 'CA', 'PB', 'MG', 'F', 'SM', 'GA', 'SN']
    dfM = df[df['Atm'].isin(metals)]
    return dfM

def eu_distance(a, b):
    """
    Calaculating distance between two atoms.
    
    Args:
        a (np.array): array of coordinates of atom1.
        b (np.array): array of coordinates of atom2.
        
    Returns:
        distance (float): distance between atom1 and atom2.
    """
    #euclidean distance
    distance = round(np.sqrt(np.sum((a - b)**2)),3)
    return distance

def angle(a, b):
    """
    Calaculating angle between two atoms.
    
    Args:
        a (np.array): array of coordinates of vector1.
        b (np.array): array of coordinates of vector2.
        
    Returns:
        theta (float): angle between vector1 and vector2.
    """
    len_a = np.sqrt(np.sum((a)**2))
    len_b = np.sqrt(np.sum((b)**2))
    dot = np.dot(a,b)
    cos = dot / (len_a*len_b)
    theta = round(math.degrees(np.arccos(cos)), 2)
    return theta

def dihedral_angle(p1, p2, p3, p4):
    """
    Calaculating dihedral angle between two planes.
    
    Args:
        p1 (np.array): array of coordinates of vector1.
        p2 (np.array): array of coordinates of vector2.
        p3 (np.array): array of coordinates of vector3.
        p4 (np.array): array of coordinates of vector4.
        
    Returns:
        phi (float): dihedral angle between the two defined planes.
    """
    q1 = p2 - p1
    q2 = p3 - p2
    q3 = p4 - p3           
    c1 = np.cross(q1,q2)
    c2 = np.cross(q2,q3)
    n1 = c1/np.sqrt(np.dot(c1,c1))
    n2 = c2/np.sqrt(np.dot(c2,c2))                    
    u1 = n2
    u3 = q2/(np.sqrt(np.dot(q2,q2)))
    u2 = np.cross(u3,u1)                    
    cos = np.dot(n1,u1)
    sin = np.dot(n1,u2)                    
    phi = round(np.degrees(-math.atan2(sin,cos)), 2)
    return phi

def phi_conversion(φ1):
    """
    Converting angle range from (0 - 360) to (-180 - +180).
    
    Args:
        φ1 (float64): angle with value ranging from (0-360).
        
    Returns:
        φ (float): angle with value ranging from (-180 - +180).
    """
    if 90 >= φ1 >= -90:
        φ = φ1
    elif φ1 > 90:
        φ = 180 - φ1
    elif φ1 < -90:
        φ = -180 - φ1
    return φ

def theta_phi(S, C1, C2, N):
    """
    Calaculating angles required to define directional criterion between interacting atoms.
    
    Args:
        S (np.array): array of coordinates of atom S.
        C1 (np.array): array of coordinates of atom C1.
        C2 (np.array): array of coordinates of atom C2.
        N (np.array): array of coordinates of atom N.
        
    Returns:
        φ (float): angle-1 to defined directional criterion.
        θ (float): angle-2 to defined directional criterion.
    """
    C = (S + C1 + C2)/3
    v1 = C - S
    v2 = N - S
    θ = angle(v1,v2)
    φ1 = dihedral_angle(C1, C, S, N)
    φ = phi_conversion(φ1)
    return φ, θ