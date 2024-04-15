import argparse


class Config:
    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser(description='MARS')
        parser.add_argument("--device", default="5", type=str)
        parser.add_argument('--subjectID', type=int, help='random seed')
        parser.add_argument('--lr', default=1e-03, type=float, help='random seed')
        parser.add_argument('--weight_decay', default=1e-03, type=float, help='random seed')
        parser.add_argument('--AClr', default=1e-04, type=float, help='random seed')
        parser.add_argument('--ACwd', default=0, type=float, help='random seed')
        parser.add_argument('--batch_size', default=72, type=int, help='random seed')
        parser.add_argument('--num_epochs', default=720, type=int, help='random seed')
        parser.add_argument('--num_epochs_pre', default=360, type=int, help='random seed')
        parser.add_argument('--seed', type=int, default=None, help='random seed')
        parser.add_argument('--gam', default=0.97, type=float, help='random seed')
        parser.add_argument('--r_gam', default=0.1, type=float, help='random seed')


        self.args = parser.parse_args()


class rayConfig:
    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser(description='MARS')
        parser.add_argument('--seed', type=int, default=None, help='random seed')
        parser.add_argument('--subjectID', type=int, help='random seed')

        self.seed = parser.parse_args().seed
        self.device = "3"
        self.subjectID = parser.parse_args().subjectID

        self.lr = 1e-03
        self.weight_decay = 1e-03
        self.AClr = 1e-04
        self.ACwd = 0
        self.batch_size = int(5)
        self.num_epochs = 40
        self.num_epochs_pre = 10


class rayConfig2a:
    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser(description='MARS')
        parser.add_argument('--seed', type=int, default=None, help='random seed')
        parser.add_argument('--subjectID', type=int, help='random seed')

        self.seed = parser.parse_args().seed
        self.device = "3"
        self.subjectID = parser.parse_args().subjectID

        self.lr = 1e-03
        self.weight_decay = 1e-03
        self.AClr = 1e-04
        self.ACwd = 0
        self.batch_size = int(72)
        self.num_epochs = 720
        self.num_epochs_pre = 360


