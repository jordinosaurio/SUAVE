## @ingroup Analyses-Aerodynamics
# Vspaero_inviscid.py
#
# Created:  Apr 2022, J. Sancho
# Modified: Apr 2022, J. Sancho
#           Apr 2022, J. Sancho

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
import SUAVE
from SUAVE.Core import Data, Units

# Local imports
from .Aerodynamics import Aerodynamics
from sklearn.gaussian_process.kernels import ExpSineSquared, RationalQuadratic, ConstantKernel, RBF, Matern
from scipy import interpolate

# Package imports
import numpy as np
import time
import matplotlib.pyplot as plt  
import sklearn
from sklearn import gaussian_process
from sklearn import neighbors
from sklearn import svm

import openvsp as vsp

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
## @ingroup Analyses-Aerodynamics
class Vspaero_inviscid(Aerodynamics):
    """This builds a surrogate and computes lift and drag using OpenVSP

    Assumptions:
    from OpenVSP model

    Source:
    None
    """   
    def __defaults__(self):
        """This sets the default values and methods for the analysis.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """ 
        self.tag = 'Vspaero_inviscid'

        self.geometry = Data()
        self.settings = Data()
        
        self.settings.vspaero_path = '/opt/OpenVSP/'

        # Conditions table, used for surrogate model training
        self.training = Data()
        self.training.angle_of_attack  = np.array([-2.,3.,8.]) * Units.deg
        self.training.Mach             = np.array([0.3,0.7,0.85])
        self.training.lift_coefficient = None
        self.training.drag_coefficient = None
        self.training_file             = None
        
        # Surrogate model
        self.surrogates = Data()
 
        
    def initialize(self):
        """Drives functions to get training samples and build a surrogate.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """                     
        # Sample training data
        self.sample_training()
                    
        # Build surrogate
        self.build_surrogate()


    def evaluate(self,state,settings,geometry):
        """Evaluates lift and drag using available surrogates.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        state.conditions.
          mach_number      [-]
          angle_of_attack  [radians]

        Outputs:
        lift      [-] CL
        drag      [-] CD

        Properties Used:
        self.surrogates.
          lift_coefficient [-] CL
          drag_coefficient [-] CD
        """  
        # Unpack
        surrogates = self.surrogates        
        conditions = state.conditions
        
        mach       = conditions.freestream.mach_number
        AoA        = conditions.aerodynamics.angle_of_attack
        lift_model = surrogates.lift_coefficient
        drag_model = surrogates.drag_coefficient
        #AR         = geometry.wings['main_wing'].aspect_ratio
        
        # Inviscid lift
        data_len = len(AoA)
        lift = np.zeros([data_len,1])
        drag = np.zeros([data_len,1])
        for ii,_ in enumerate(AoA):
            lift[ii] = lift_model([AoA[ii,0],mach[ii,0]])
            drag[ii] = drag_model([AoA[ii,0],mach[ii,0]])
        
        conditions.aerodynamics.lift_coefficient                               = lift
        conditions.aerodynamics.lift_breakdown                                 = Data()
        conditions.aerodynamics.lift_breakdown.compressible_wings              = Data()
        conditions.aerodynamics.lift_breakdown.inviscid_wings                  = Data()
        conditions.aerodynamics.lift_breakdown.total                           = lift        
        # conditions.aerodynamics.lift_breakdown.compressible_wings['main_wing'] = lift # currently using vehicle drag for wing     
        # conditions.aerodynamics.lift_breakdown.inviscid_wings['main_wing']     = lift # currently using vehicle drag for wing  
                                                                           
        conditions.aerodynamics.drag_coefficient                                    = drag       
        conditions.aerodynamics.drag_breakdown.induced                              = Data()
        conditions.aerodynamics.drag_breakdown.induced.total                        = drag
        conditions.aerodynamics.drag_breakdown.induced.inviscid                     = drag
        conditions.aerodynamics.drag_breakdown.induced.inviscid_wings               = Data()
        # conditions.aerodynamics.drag_breakdown.induced.inviscid_wings['main_wing']  = drag
        conditions.aerodynamics.drag_breakdown.untrimmed                            = drag
        conditions.aerodynamics.drag_breakdown.miscellaneous                       = Data()
        
        return lift, drag


    def sample_training(self):
        """Call methods to run vspaero for sample point evaluation.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        see properties used

        Outputs:
        self.training.
          coefficients     [-] CL and CD
          grid_points      [radians,-] angles of attack and mach numbers 

        Properties Used:
        self.geometry.tag  <string>
        self.training.     
          angle_of_attack  [radians]
          Mach             [-]
        self.training_file (optional - file containing previous AVL data)
        """               
        # Unpack
        geometry = self.geometry
        settings = self.settings
        training = self.training
        
        AoA  = training.angle_of_attack
        mach = training.Mach 
        CL   = np.zeros([len(AoA)*len(mach),1])
        CD   = np.zeros([len(AoA)*len(mach),1])

        # Condition input, local, do not keep (k is used to avoid confusion)
        konditions              = Data()
        konditions.aerodynamics = Data()

        if self.training_file is None:
            # Calculate aerodynamics for table
            table_size = len(AoA)*len(mach)
            xy = np.zeros([table_size,2])
            count = 0
            time0 = time.time()
            for i,_ in enumerate(AoA):
                for j,_ in enumerate(mach):
                    
                    xy[count,:] = np.array([AoA[i],mach[j]])
                    # Set training conditions
                    konditions.aerodynamics.angle_of_attack = AoA[i]
                    konditions.aerodynamics.mach            = mach[j]
                    
                    CL[count],CD[count] = call_openvsp(konditions, settings, geometry)
                    count += 1
            
            time1 = time.time()
            
            print('The total elapsed time to run vspaero: '+ str(time1-time0) + '  Seconds')
        else:
            data_array = np.loadtxt(self.training_file)
            xy         = data_array[:,0:2]
            CL         = data_array[:,2:3]
            CD         = data_array[:,3:4]

        # Save the data
        np.savetxt(geometry.tag+'_data.txt',np.hstack([xy,CL,CD]),fmt='%10.8f',header='AoA Mach CL CD')

        # Store training data
        training.coefficients = np.hstack([CL,CD])
        training.grid_points  = xy
        

        return

    def build_surrogate(self):
        """Builds a surrogate based on sample evalations using a Guassian process.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        self.training.
          coefficients     [-] CL and CD
          grid_points      [radians,-] angles of attack and mach numbers 

        Outputs:
        self.surrogates.
          lift_coefficient <Guassian process surrogate>
          drag_coefficient <Guassian process surrogate>

        Properties Used:
        No others
        """  
        # Unpack data
        training  = self.training
        AoA_data  = training.angle_of_attack
        mach_data = training.Mach
        CL_data   = training.coefficients[:,0]
        CD_data   = training.coefficients[:,1]
        xy        = training.grid_points 
        
              
        # Gaussian Process New
        # gp_kernel_ES = ExpSineSquared(length_scale=1.0, periodicity=1.0, length_scale_bounds=(1e-5,1e5), periodicity_bounds=(1e-5,1e5))
        # regr_cl = gaussian_process.GaussianProcessRegressor(kernel=gp_kernel_ES)
        # regr_cd = gaussian_process.GaussianProcessRegressor(kernel=gp_kernel_ES)
        
        # gp_kernel = Matern()
        # regr_cl = gaussian_process.GaussianProcessRegressor(kernel=gp_kernel)
        # regr_cd = gaussian_process.GaussianProcessRegressor(kernel=gp_kernel)        
        # cl_surrogate = regr_cl.fit(xy, CL_data)
        # cd_surrogate = regr_cd.fit(xy, CD_data)  
        
        # KNN
        # regr_cl = neighbors.KNeighborsRegressor(n_neighbors=3,weights='distance')
        # regr_cd = neighbors.KNeighborsRegressor(n_neighbors=3,weights='distance')
        # cl_surrogate = regr_cl.fit(xy, CL_data)
        # cd_surrogate = regr_cd.fit(xy, CD_data)  
        
        # SVR
        #regr_cl = svm.SVR(C=500.)
        #regr_cd = svm.SVR()
        #cl_surrogate = regr_cl.fit(xy, CL_data)
        #cd_surrogate = regr_cd.fit(xy, CD_data)
        
        #interp2D
        
        z = CL_data.reshape(len(AoA_data),len(mach_data))
        cl_surrogate = interpolate.RegularGridInterpolator((AoA_data,mach_data),z, bounds_error=False, fill_value=None)
        z = CD_data.reshape(len(AoA_data),len(mach_data))
        cd_surrogate = interpolate.RegularGridInterpolator((AoA_data,mach_data),z, bounds_error=False, fill_value=None)
        
        
        self.surrogates.lift_coefficient = cl_surrogate
        self.surrogates.drag_coefficient = cd_surrogate
        
        # Standard subsonic test case
        AoA_points = np.linspace(-10.0,20.0,30)*Units.deg
        mach_points = np.linspace(0.0,1.0,10)
        
        AoA_mesh,mach_mesh = np.meshgrid(AoA_points,mach_points,indexing='ij')
        
        table_size = len(AoA_points)*len(mach_points)
        CL_points = np.zeros(table_size)
        CD_points = np.zeros(table_size)
        count = 0
        for i,_ in enumerate(AoA_points):
            for j,_ in enumerate(mach_points):
                CL_points[count] = cl_surrogate([AoA_points[i],mach_points[j]])
                CD_points[count] = cd_surrogate([AoA_points[i],mach_points[j]])
                count += 1
        
        # CL_sur = np.zeros(np.shape(AoA_mesh))
        # CD_sur = np.zeros(np.shape(AoA_mesh))        
        
        # for jj in range(len(AoA_points)):
        #     for ii in range(len(mach_points)):
        #         CL_sur[ii,jj] = cl_surrogate.predict([np.array([AoA_mesh[ii,jj],mach_mesh[ii,jj]])])
        #         CD_sur[ii,jj] = cd_surrogate.predict([np.array([AoA_mesh[ii,jj],mach_mesh[ii,jj]])])  #sklearn fix
        
        CL_sur = CL_points.reshape(len(AoA_points),len(mach_points))
        CD_sur = CD_points.reshape(len(AoA_points),len(mach_points))

        fig = plt.figure('Coefficient of Lift Surrogate Plot')    
        plt_handle = plt.contourf(AoA_mesh/Units.deg,mach_mesh,CL_sur,levels=None)
        #plt.clabel(plt_handle, inline=1, fontsize=10)
        cbar = plt.colorbar()
        plt.scatter(xy[:,0]/Units.deg,xy[:,1])
        plt.xlabel('Angle of Attack (deg)')
        plt.ylabel('Mach Number')
        cbar.ax.set_ylabel('Coefficient of Lift')

        # Stub for plotting drag if implemented:

        #plt.contourf(AoA_mesh/Units.deg,mach_mesh,CD_sur,levels=None)
        #plt.colorbar()
        #plt.xlabel('Angle of Attack (deg)')
        #plt.ylabel('Mach Number')   
        
        plt.show() 

        return



# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------

def call_openvsp(conditions,settings,geometry):
    """Calculates lift and drag using openvsp

    Assumptions:
    None

    Source:
    openvsp

    Inputs:
    conditions.
      mach_number        [-]
      angle_of_attack    [radians]
    settings.
      maximum_iterations [-]
    geometry.
      tag
      reference_area     [m^2]

    Outputs:
    CL                   [-]
    CD                   [-]

    Properties Used:
    N/A
    """      

    tag            = geometry.tag
    vspaero_path   = settings.vspaero_path
    vsp3_path      = tag + '.vsp3'
    
    vsp.SetVSPAEROPath(vspaero_path)

    """clear VSPmodel"""
    vsp.ClearVSPModel()

    """read file"""
    vsp.ReadVSPFile(vsp3_path)
    
    """update conditions"""
    parm_container_name = 'Default'
    parm_container_id = vsp.FindContainersWithName(parm_container_name)[0]
    
    group_name = 'VSPAERO'
    
    parm_name = 'AlphaStart'
    parm_id = vsp.FindParm( parm_container_id, parm_name,  group_name)
    param_val = conditions.aerodynamics.angle_of_attack / Units.deg
    vsp.SetParmVal(parm_id, param_val)
    vsp.Update()
    
    parm_name = 'MachStart'
    parm_id = vsp.FindParm( parm_container_id, parm_name,  group_name)
    param_val = conditions.aerodynamics.mach
    vsp.SetParmVal(parm_id, param_val)
    vsp.Update()
    
    """compute degenerate geometry for vl"""
    analysisName = "VSPAEROComputeGeometry"
    vsp.SetAnalysisInputDefaults(analysisName)
    resId = vsp.ExecAnalysis(analysisName)
    vsp.Update()
    
    """prepare analysis"""
    analysisName = "VSPAEROSweep"
    vsp.SetAnalysisInputDefaults(analysisName)
    
    """execute analysis"""
    resId = vsp.ExecAnalysis(analysisName)
    polar_id = vsp.FindResultsID('VSPAERO_History')
    
    names = vsp.GetAllDataNames(polar_id)
    name = vsp.GetResultsName(polar_id)

    data = []
    for name in names:
        type = vsp.GetResultsType(polar_id, name)
        d = []
        if type == vsp.INT_DATA:
            d = list(vsp.GetIntResults(polar_id, name))
        elif type == vsp.STRING_DATA:
            d = list(vsp.GetStringResults(polar_id, name))
        elif type == vsp.DOUBLE_DATA:
            d = list(vsp.GetDoubleResults(polar_id, name))
        elif type == vsp.DOUBLE_MATRIX_DATA:
            d = vsp.convert_double_tuple_to_list_matrix(vsp.GetDoubleMatResults(polar_id, name))
        elif type == vsp.VEC3D_DATA:
            d = vsp.convert_vec3d_array_to_list_matrix(vsp.GetVec3dResults(polar_id, name))

        data.append([name,d[-1]])    
    
    CL = data[10][1]
    CD = data[5][1]
    
    vsp.DeleteAllResults()
        
    return CL, CD
