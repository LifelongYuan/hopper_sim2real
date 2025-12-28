import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def rand_from_range(range,size):
    low,high = range
    interval = (high-low)/2
    c = (high+low)/2
    # to [-1,1]
    sample = (np.random.rand((size))*2-1)*interval +c
    if size==1:
        sample = sample[0]
    return sample

def forward_kinematics():
    pass


def imu2body(): # imu vector to body 
    return np.array([[0,0,-1],[-1,0,0],[0,1,0]])

def foot2body():
    return np.array([[0,1,0],[1,0,0],[0,0,-1]])

def realbody2simbody():
    mat = R.from_euler("xyz",[0,0,0]).as_matrix()
    return mat

def fk_simplified(th1,th2,d):
    """
    d: always negative
    """
    t1 = np.array([[1,0,0,0],[0,np.cos(th1),-np.sin(th1),0],[0,-np.sin(th1),np.cos(th1),0],[0,0,0,1]])
    t2 = np.array([[np.cos(th2),0,np.sin(th2),0],[0,1,0,0],[-np.sin(th2),0,np.cos(th2),0],[0,0,0,1]])
    t3 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,d],[0,0,0,1]])
    t = np.dot(t1,t2).dot(t3)
    foot2body = np.array([[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]])
    t = foot2body.dot(t)
    print(t)
    return t.T[3][0:3]

def fk_simplified_torch(th1, th2, d):
    """
    d: always negative
    th1, th2, d: Expected to have shape [batch_num, 1]
    """
    # t1 transformation matrix for each batch
    t1 = torch.stack([
        torch.ones_like(th1).squeeze(), torch.zeros_like(th1).squeeze(), torch.zeros_like(th1).squeeze(), torch.zeros_like(th1).squeeze(),
        torch.zeros_like(th1).squeeze(), torch.cos(th1).squeeze(), -torch.sin(th1).squeeze(), torch.zeros_like(th1).squeeze(),
        torch.zeros_like(th1).squeeze(), -torch.sin(th1).squeeze(), torch.cos(th1).squeeze(), torch.zeros_like(th1).squeeze(),
        torch.zeros_like(th1).squeeze(), torch.zeros_like(th1).squeeze(), torch.zeros_like(th1).squeeze(), torch.ones_like(th1).squeeze()
    ], dim=1).reshape(-1, 4, 4)  # [batch_num, 4, 4]
    
    # t2 transformation matrix for each batch
    t2 = torch.stack([
        torch.cos(th2).squeeze(), torch.zeros_like(th2).squeeze(), torch.sin(th2).squeeze(), torch.zeros_like(th2).squeeze(),
        torch.zeros_like(th2).squeeze(), torch.ones_like(th2).squeeze(), torch.zeros_like(th2).squeeze(), torch.zeros_like(th2).squeeze(),
        -torch.sin(th2).squeeze(), torch.zeros_like(th2).squeeze(), torch.cos(th2).squeeze(), torch.zeros_like(th2).squeeze(),
        torch.zeros_like(th2).squeeze(), torch.zeros_like(th2).squeeze(), torch.zeros_like(th2).squeeze(), torch.ones_like(th2).squeeze()
    ], dim=1).reshape(-1, 4, 4)  # [batch_num, 4, 4]
    
    # t3 transformation matrix for each batch
    t3 = torch.stack([
        torch.ones_like(d).squeeze(), torch.zeros_like(d).squeeze(), torch.zeros_like(d).squeeze(), torch.zeros_like(d).squeeze(),
        torch.zeros_like(d).squeeze(), torch.ones_like(d).squeeze(), torch.zeros_like(d).squeeze(), torch.zeros_like(d).squeeze(),
        torch.zeros_like(d).squeeze(), torch.zeros_like(d).squeeze(), torch.ones_like(d).squeeze(), d.squeeze(),
        torch.zeros_like(d).squeeze(), torch.zeros_like(d).squeeze(), torch.zeros_like(d).squeeze(), torch.ones_like(d).squeeze()
    ], dim=1).reshape(-1, 4, 4)  # [batch_num, 4, 4]
    
    # Matrix multiplication: T = t1 * t2 * t3
    t = torch.matmul(torch.matmul(t1, t2), t3)  # [batch_num, 4, 4]
    
    # Extract the translation vector (last column, first 3 rows)
    translation_vector = t[:, :3, 3]  # [batch_num, 3]
    
    return translation_vector


def jacobian(th1,th2,d):
    n1 = np.array([1,0,0]).T
    n2 = np.array([0,np.cos(th1),np.sin(th1)]).T
    n3 = np.array([np.sin(th2),-np.sin(th1)*np.cos(th2),np.cos(th1)*np.cos(th2)]).T
    r = np.array([np.sin(th2)*d,-np.sin(th1)*np.cos(th2)*d,np.cos(th1)*np.cos(th2)*d]).T
    col1 = np.cross(n1,r)
    col2 = np.cross(n2,r)
    col3 = n3
    print("col1",col1)
    J_v = np.column_stack([col1,col2,col3])
    print("n3",n3)
    J_w = np.column_stack([n1,n2,n3*0])
    print("J_v",J_v)
    print("J_w",J_w)
    return J_v,J_w

def jacobian_tensor(th1, th2, d,n1):
    # Define unit vector n1 and replicate it across the batch dimension    
    # Define vectors n2 and n3 with batch dimension
    n2 = torch.cat([torch.zeros_like(th1), torch.cos(th1), torch.sin(th1)], dim=1)  # [batch_num, 3]
    n3 = torch.cat([torch.sin(th2), -torch.sin(th1) * torch.cos(th2), torch.cos(th1) * torch.cos(th2)], dim=1)  # [batch_num, 3]
    
    # Compute r vector with batch dimension
    r = torch.cat([torch.sin(th2) * d, -torch.sin(th1) * torch.cos(th2) * d, torch.cos(th1) * torch.cos(th2) * d], dim=1)  # [batch_num, 3]
    
    # Compute columns of the Jacobian matrix J_v
    col1 = torch.cross(n1, r, dim=1)  # [batch_num, 3]
    col2 = torch.cross(n2, r, dim=1)  # [batch_num, 3]
    col3 = n3  # [batch_num, 3]
    
    # Stack columns to form the Jacobian matrices
    J_v = torch.stack([col1, col2, col3], dim=2)  # [batch_num, 3, 3]
    J_w = torch.stack([n1, n2, torch.zeros_like(n3)], dim=2)  # [batch_num, 3, 3]
    
    return J_v, J_w
    
def ik_simplified(p_foot):
    r = -np.sqrt(p_foot[0]**2+p_foot[1]**2+p_foot[2]**2)
    roll = np.arcsin(p_foot[1]/np.sqrt(p_foot[2]**2+p_foot[1]**2))
    pitch = np.arcsin( p_foot[0]/r)
    return [roll,pitch,r]

def get_translation_jacobian(th1,th2,d):
    pass

def inverse_kinematics(x, D, d, r, upLim, lowLim):
    theta = np.zeros(3)
    check = np.zeros(3)
    noSol = np.zeros(3)
    
    R2 = np.array([[np.cos(np.radians(120)), -np.sin(np.radians(120)), 0],
                   [np.sin(np.radians(120)), np.cos(np.radians(120)), 0],
                   [0, 0, 1]])
    R3 = np.array([[np.cos(np.radians(240)), -np.sin(np.radians(240)), 0],
                   [np.sin(np.radians(240)), np.cos(np.radians(240)), 0],
                   [0, 0, 1]])

    temp = ((r - x[1])**2 / 2 + D**2 / 2 - d**2 / 2 + x[0]**2 / 2 + x[2]**2 / 2) / (D * np.sqrt((r - x[1])**2 + x[2]**2))
    if abs(temp) > 1:
        noSol[0] = 1
    else:
        theta[0] = - np.arccos(((r - x[1])**2 / 2 + D**2 / 2 - d**2 / 2 + x[0]**2 / 2 + x[2]**2 / 2) / (D * np.sqrt((r - x[1])**2 + x[2]**2))) + np.arctan2(x[2], x[1] - r)
        knee1 = np.array([0, r, 0]) + np.array([0, D * np.cos(theta[0]), D * np.sin(theta[0])])
        check[0] = np.linalg.norm(x - knee1) - d
        if theta[0] > upLim:
            if theta[0] - 2 * np.pi < lowLim:
                noSol[0] = 2
        elif theta[0] < lowLim:
            if theta[0] + 2 * np.pi > upLim:
                noSol[0] = 2

    temp = ((r + x[1] / 2 + np.sqrt(3) * x[0] / 2)**2 / 2 + D**2 / 2 - d**2 / 2 + (x[0] / 2 - np.sqrt(3) * x[1] / 2)**2 / 2 + x[2]**2 / 2) / (D * np.sqrt((r + x[1] / 2 + np.sqrt(3) * x[0] / 2)**2 + x[2]**2))
    if abs(temp) > 1:
        noSol[1] = 1
    else:
        theta[1] = np.arctan2(x[2], - r - x[1] / 2 - np.sqrt(3) * x[0] / 2) - np.arccos(((r + x[1] / 2 + np.sqrt(3) * x[0] / 2)**2 / 2 + D**2 / 2 - d**2 / 2 + (x[0] / 2 - np.sqrt(3) * x[1] / 2)**2 / 2 + x[2]**2 / 2) / (D * np.sqrt((r + x[1] / 2 + np.sqrt(3) * x[0] / 2)**2 + x[2]**2)))
        knee2 = R2.dot(np.array([0, r, 0]) + np.array([0, D * np.cos(theta[1]), D * np.sin(theta[1])]))
        check[1] = np.linalg.norm(x - knee2) - d
        if theta[1] > upLim:
            if theta[1] - 2 * np.pi < lowLim:
                noSol[1] = 2
        elif theta[1] < lowLim:
            if theta[1] + 2 * np.pi > upLim:
                noSol[1] = 2

    temp = ((r + x[1] / 2 - np.sqrt(3) * x[0] / 2)**2 / 2 + D**2 / 2 - d**2 / 2 + (x[0] / 2 + np.sqrt(3) * x[1] / 2)**2 / 2 + x[2]**2 / 2) / (D * np.sqrt((r + x[1] / 2 - np.sqrt(3) * x[0] / 2)**2 + x[2]**2))
    if abs(temp) > 1:
        noSol[2] = 1
    else:
        theta[2] = np.arctan2(x[2], np.sqrt(3) * x[0] / 2 - x[1] / 2 - r) - np.arccos(((r + x[1] / 2 - np.sqrt(3) * x[0] / 2)**2 / 2 + D**2 / 2 - d**2 / 2 + (x[0] / 2 + np.sqrt(3) * x[1] / 2)**2 / 2 + x[2]**2 / 2) / (D * np.sqrt((r + x[1] / 2 - np.sqrt(3) * x[0] / 2)**2 + x[2]**2)))
        knee3 = R3.dot(np.array([0, r, 0]) + np.array([0, D * np.cos(theta[2]), D * np.sin(theta[2])]))
        check[2] = np.linalg.norm(x - knee3) - d
        if theta[2] > upLim:
            if theta[2] - 2 * np.pi < lowLim:
                noSol[2] = 2
        elif theta[2] < lowLim:
            if theta[2] + 2 * np.pi > upLim:
                noSol[2] = 2

    return theta, check, noSol

if __name__=="__main__":
    import time 
    range = [-2,3]
    sample = rand_from_range(range,3)
    # print(sample)
    D = 0.158
    d = 0.398
    r = 0.02424
    upLim = 80*np.pi/180
    lowLim = -60*np.pi/180

    t = fk_simplified(0,0.,-0.5)
    print("t",t)
    p = t
    theta, check, noSol=inverse_kinematics(p,D,d,r,upLim,lowLim)
    re = ik_simplified(p)
    jacobian(1.57,0,0.5)
    # print("foot_pos",p)
    # print("simplified_joint",re)
    # print("ori_model_joint",theta)
    p = [0.05,0.05,0.4]
    inverse_kinematics(p,D,d,r,upLim,lowLim)

    batch_num = 2048
    now = time.time()
    device = "cuda:0"
    th1 = torch.zeros(batch_num, 1,device=device) + 1.57
    th2 = torch.zeros(batch_num, 1,device=device) + 0.38
    d = torch.zeros(batch_num, 1,device=device) + 0.5
    n1 = torch.tensor([1.0, 0.0, 0.0],device=device).unsqueeze(0).expand(th1.size(0), -1)  # [batch_num, 3]

    J_v,J_w = jacobian_tensor(th1,th2,d,n1)
    # print("spent",time.time()-now)
    now = time.time()
    J_v,J_w = jacobian_tensor(th1,th2,d,n1)
    # print("jit_spent",time.time()-now)
    # print(J_v)
    
    now = time.time()
    # print("th1",th1)
    # print("th2",th2)
    # print("d",d)

    # foot_pos = fk_simplified_torch(th1,th2,d)
    # print("forward_k",time.time()-now)
    # print("torch_foot_pos",foot_pos[0])