class Metadata:
    ''' 
    Metadata:
        Contain everything about an image
        - Mask
        - Image
        - Description
    '''
    def __init__(self, meta, dataset, img, mask):
        self.meta = meta
        self.dataset = dataset
        self.img = img
        self.type = meta["clinical"]["benign_malignant"]  # malignant , benign
        self.mask = mask
