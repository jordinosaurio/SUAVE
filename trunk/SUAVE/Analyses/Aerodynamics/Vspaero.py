## @ingroup Analyses-Aerodynamics
# SU2_Euler.py
#
# Created:  Apr 2022, J. Sancho
# Modified: Apr 2022, J. Sancho
#           Apr 2022, J. Sancho

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from .Markup import Markup
from SUAVE.Analyses import Process
import numpy as np


# The aero methods
from SUAVE.Methods.Aerodynamics.Common import Fidelity_Zero as Common
from .Process_Geometry import Process_Geometry
from SUAVE.Analyses.Aerodynamics.Vspaero_inviscid import Vspaero_inviscid

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
## @ingroup Analyses-Aerodynamics
class Vspaero(Markup):
    """This uses vspaero to compute lift.

    Assumptions:
    Subsonic

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
        self.tag    = 'Vspaero_markup'       
    
        # Correction factors
        settings = self.settings
        settings.trim_drag_correction_factor        = 1.02
        # settings.wing_parasite_drag_form_factor     = None
        # settings.fuselage_parasite_drag_form_factor = None
        # settings.oswald_efficiency_factor           = None
        # settings.span_efficiency                    = None
        # settings.viscous_lift_dependent_drag_factor = None
        settings.drag_coefficient_increment         = 0.0
        settings.spoiler_drag_increment             = 0.0 
        # settings.maximum_lift_coefficient           = np.inf 
        # settings.recalculate_total_wetted_area      = False
        
        # Build the evaluation process
        compute = self.process.compute

        # Run Vspaero to determine lift
        compute.lift = Process()
        compute.lift.inviscid                      = Vspaero_inviscid()
        compute.lift.total                         = Common.Lift.aircraft_total
        
        # Do a traditional drag buildup
        compute.drag = Process()
        # compute.drag.parasite                      = Process()
        # compute.drag.parasite.wings                = Process_Geometry('wings')
        # compute.drag.parasite.wings.wing           = Common.Drag.parasite_drag_wing 
        # compute.drag.parasite.fuselages            = Process_Geometry('fuselages')
        # compute.drag.parasite.fuselages.fuselage   = Common.Drag.parasite_drag_fuselage
        # compute.drag.parasite.nacelles             = Process_Geometry('nacelles')
        # compute.drag.parasite.nacelles.nacelle     = Common.Drag.parasite_drag_nacelle 
        # compute.drag.parasite.pylons               = Common.Drag.parasite_drag_pylon
        # compute.drag.parasite.total                = Common.Drag.parasite_total
        # compute.drag.induced                       = Common.Drag.induced_drag_aircraft
        # compute.drag.compressibility               = Process()
        # compute.drag.compressibility.wings         = Process_Geometry('wings')
        # compute.drag.compressibility.wings.wing    = Common.Drag.compressibility_drag_wing
        # compute.drag.compressibility.total         = Common.Drag.compressibility_drag_wing_total        
        # compute.drag.miscellaneous                 = Common.Drag.miscellaneous_drag_aircraft_ESDU
        # compute.drag.untrimmed                     = Common.Drag.untrimmed
        compute.drag.trim                          = Common.Drag.trim
        compute.drag.spoiler                       = Common.Drag.spoiler_drag
        compute.drag.total                         = Common.Drag.total_aircraft
        
        
    def initialize(self):
        """Initializes the surrogate needed for Vspaero

        Assumptions:
        Vehicle is available in OpenVSP
        
        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        self.geometry.tag               <string> (geometry is also set as part of the lift process)
        self.process.compute.lift.
          inviscid.training_file        (optional - determines if new runs are necessary)
        self.settings.
          half_mesh_flag                <boolean> Determines if a symmetry plane is used
          vsp_mesh_growth_ratio         [-] Determines how the mesh grows
          vsp_mesh_growth_limiting_flag <boolean> Determines if 3D growth limiting is used
        """         
        super(Vspaero, self).initialize()
        self.process.compute.lift.inviscid.geometry = self.geometry
        
        tag = self.geometry.tag
        
        # Generate the surrogate
        self.process.compute.lift.inviscid.initialize()
        
    finalize = initialize