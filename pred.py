from model import *
from data import *
from main import *

testGene = testGenerator("data1/test")

results = model.predict_generator(testGene,10,verbose=1)
saveResult("data1/test1",results)