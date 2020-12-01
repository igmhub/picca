"""This module define several functions and variables used throughout the
package"""
import importlib

ABSORBER_IGM = {
    "Halpha"      : 6562.8,
    "Hbeta"       : 4862.68,
    "MgI(2853)"   : 2852.96,
    "MgII(2804)"  : 2803.5324,
    "MgII(2796)"  : 2796.3511,
    "FeII(2600)"  : 2600.1724835,
    "FeII(2587)"  : 2586.6495659,
    "MnII(2577)"  : 2576.877,
    "FeII(2383)"  : 2382.7641781,
    "FeII(2374)"  : 2374.4603294,
    "FeII(2344)"  : 2344.2129601,
    "AlIII(1863)" : 1862.79113,
    "AlIII(1855)" : 1854.71829,
    "AlII(1671)"  : 1670.7886,
    "FeII(1608)"  : 1608.4511,
    "CIV(1551)"   : 1550.77845,
    "CIV(eff)"    : 1549.06,
    "CIV(1548)"   : 1548.2049,
    "SiII(1527)"  : 1526.70698,
    "NiII(1455)"  : 1454.842,
    "SiIV(1403)"  : 1402.77291,
    "SiIV(1394)"  : 1393.76018,
    "NiII(1370)"  : 1370.132,
    "CII(1335)"   : 1334.5323,
    "NiII(1317)"  : 1317.217,
    "SiII(1304)"  : 1304.3702,
    "OI(1302)"    : 1302.1685,
    "SiII(1260)"  : 1260.4221,
    "SII(1254)"   : 1253.811,
    "SII(1251)"   : 1250.584,
    "NV(1243)"    : 1242.804,
    "NV(1239)"    : 1238.821,
    "LYA"         : 1215.67,
    "SiIII(1207)" : 1206.500,
    "NI(1200)"    : 1200.,
    "SiII(1193)"  : 1193.2897,
    "SiII(1190)"  : 1190.4158,
    "PII(1153)"   : 1152.818,
    "FeII(1145)"  : 1144.9379,
    "FeII(1143)"  : 1143.2260,
    "NI(1134)"    : 1134.4149,
    "FeII(1125)"  : 1125.4477,
    "FeIII(1123)" : 1122.526,
    "FeII(1097)"  : 1096.8769,
    "NII(1084)"   : 1083.990,
    "FeII(1082)"  : 1081.8748,
    "FeII(1063)"  : 1063.002,
    "OI(1039)"    : 1039.230,
    "OVI(1038)"   : 1037.613,
    "CII(1037)"   : 1036.7909,
    "OVI(1032)"   : 1031.912,
    "LYB"         : 1025.72,
    "SiII(1021)"  : 1020.6989,
    "SIII(1013)"  : 1012.502,
    "SiII(990)"   : 989.8731,
    "OI(989)"     : 988.7,
    "CIII(977)"   : 977.020,
    "LY3"         : 972.537,
    "LY4"         : 949.7431,
    "LY5"         : 937.8035,
    "LY6"         : 930.7483,
    "LY7"         : 926.2257,
    "LY8"         : 923.1504,
    "LY9"         : 920.9631,
    "LY10"        : 919.3514,
}

def class_from_string(class_name, module_name):
    """Returns a class from a string. The class must be saved in a module
    under picca.delta_extraction with the same name as the class but
    lowercase and with and underscore. For example class 'MyClass' should
    be in module picca.delta_extraction.my_class

    Arguments
    ---------
    class_name: str
    Name of the class to load

    module_name: str
    Name of the module containing the class

    Returns
    -------
    The loaded class

    Raises
    ------
    ImportError if module cannot be loaded
    AttributeError if class cannot be found
    """
    # load module
    module_object = importlib.import_module(module_name)
    # get the class, will raise
    class_object = getattr(module_object, class_name)
    return class_object
