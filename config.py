import os

DATA_DIR = "Animal-10-split"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
STEP_SIZE = 10
GAMMA = 0.1
DROPOUT_RATE = 0.25

NUM_CLASSES = 10