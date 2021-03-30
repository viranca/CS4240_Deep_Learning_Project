from gymware import Warehouse
import matplotlib.pyplot as plt
import numpy as np
import yaml
import random
random.seed(1)


warehouse = Warehouse()
warehouse.render()
(warehouse.step(0))
(warehouse.step(0))
(warehouse.step(0))
(warehouse.step(2))
(warehouse.step(2))
print(warehouse.step(2))

