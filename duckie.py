import yaml
import numpy as np
with open("gym-duckietown/src/gym_duckietown/maps/duckies.yaml", 'r') as file:
    data = yaml.safe_load(file)

DUCKIES = 25


def random_duckie():
    random_pos = [round(np.random.uniform(0,7), 2), round(np.random.uniform(0,7), 2)]
    random_rotation = np.random.randint(0, 360) 
    return {'kind': 'duckie', 'pos': random_pos, 'rotate': random_rotation, 'height': 0.06}
    

random_duckies = [random_duckie() for i in range(DUCKIES)]
# print(random_duckies)

data['objects'] = random_duckies

with open("gym-duckietown/src/gym_duckietown/maps/duckies-rand.yaml", 'w') as file:
    yaml.dump(data, file)
# print(data['objects'])

