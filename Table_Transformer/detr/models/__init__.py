##%%%%%%%%  Author       :: Ankit Malviya & Kamanksha P. Dubey   
##%%%%%%%%  Roll Number  :: PhD2201101011 & PhD2201101001
##%%%%%%%%  Course       :: PhD
##%%%%%%%%  Department   :: Computer Science & Engineering

from .detr import build
from .detr_multi import build as build_multi


def build_model(args):
    return build(args)

def build_model_multi(args):
    return build_multi(args)