import numpy

def normalize(x):
  return (x-x.min()) / (x.max()-x.min())