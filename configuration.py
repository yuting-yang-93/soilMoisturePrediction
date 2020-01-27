import datetime as dt


class ModelConfiguration:

    def __init__(self):
        super(ModelConfiguration, self).__init__()

        # set the date
        self.date_start = dt.datetime(2008, 1, 1)
        self.date_end = dt.datetime(2019, 10, 24)

        self.need_depth = [5, 10, 20, 50, 100]  # which depths of soil moisture you need
        self.timeSeriesUnit = 'H'  # 'H' means hour, 'D' means day
        # Varaibles
        # - Physical soil properties
        #     - BLDFIE: Bulk Density ('kg/cubic_m')
        #     - CLYPPT: Caly Content (percent)
        #     - CRFVOL: Coarse Fragements Volumetric (%) (percent)
        #     - SLTPPT: Slit Content (percent)
        #     - SNDPPT: Sand Content (percent)
        # - Chemical Soil Properties
        #     - CECSOL: Cation Excahnge capacit of soil (cmol / kg)
        #     - ORCDRC: Soil organic carbon content (g / kg)
        #     - PHIHOX: Soil pH * 10 in H20 (Index*10)
        #     - PHIKCL: Soil pH * 10 in KCL (Index*10)
        self.soilProperties = ['BLDFIE', 'CLYPPT', 'CRFVOL', 'SLTPPT', 'SNDPPT', 'CECSOL', 'ORCDRC', 'PHIHOX', 'PHIKCL']
        self.classProperties = ['TAXNWRB', 'TAXOUSDA']
        self.testStations = ['Selma', 'LittleRiver', 'ErosDataCenter', 'MountainHome', 'BlueCreek', 'Tunica', 'UAPBPointRemove', 'EagleLake', 'Perthshire', 'BraggFarm', 'PowellGardens']
        self.inputVar = ['station', 'sm_5', 'p', 'st_5', 'at', 'TAXNWRB_Class1', 'TAXOUSDA_Class1', 'BLDFIE_sl1',
                         'CLYPPT_sl1', 'CRFVOL_sl1', 'SLTPPT_sl1', 'SNDPPT_sl1', 'CECSOL_sl1', 'ORCDRC_sl1',
                         'PHIHOX_sl1', 'PHIKCL_sl1']
        # then do not use this station
        self.threshold_missingValue = 0.5
        # 'TAXNWRB_Class1', 'TAXNWRB_Class2', 'TAXNWRB_Class3',
        # 'TAXOUSDA_Class1', 'TAXOUSDA_Class2', 'TAXOUSDA_Class3'
        # self.soilClassVar = 'TAXNWRB_Class1'
        # if soil types has at least threshold_stations, then set as
        # self.threshold_stations = 5

        self.warmup_steps = 30

        # for first preprocessing: outlier detection by quantile
        self.numeric_columns = ['sm_5','sm_10','sm_20','sm_50','sm_100','st_5','st_10','st_20','st_50','st_100','at']
        # for second preprocessing: outlier detection by Isolation Forest
        self.to_IForest_columns = ['sm_5','sm_20','sm_50','sm_100', 'p', 'at','st_5']
  
  
        self.EWLR_K = 24
        self.EWLR_A = [0.94, 0.99]
        self.timesteps = 24 * 7
        self.validation_split = 0.15

        self.overlappedSize = 48