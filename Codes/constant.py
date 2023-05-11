#training dataset path
TRAIN_FOLDER = '/data/cylin/nl/Data/DRC-D/training'

#testing dataset path
TEST_FOLDER = '/data/cylin/nl/Data/DRC-D/testing'

#testing dataset path for other datasets
TEST_OTHER_FOLDER = '../Other_dataset/'

#GPU index
GPU = '4'

#batch size for training
TRAIN_BATCH_SIZE = 4

#batch size for testing
TEST_BATCH_SIZE = 1

#num of iters   
ITERATIONS = 150000 

# checkpoints path
SNAPSHOT_DIR = "./checkpoints"

#sumary path
SUMMARY_DIR = "./summary"

# define the mesh resolution
GRID_W = 8
GRID_H = 6