import os
import numpy as np
import matplotlib.pyplot as plt
import imageio



def saveGif(matrixList,gifFileName,clearFrames=True):
    filenames=[]
    for i,m in enumerate(matrixList):
        # plot the line chart
        #plt.plot(y[:i])
        plt.imshow(np.squeeze(m), extent=[-180,180,-90,90]) #minx maxx miny maxy
        
        # create file name and append it to a list
        filename = f'{gifFileName}_{i}.png'
        filenames.append(filename)
        plt.title(f"Day {int(np.floor(i/24))+1} hour {i%24:02d}")
        # save frame
        plt.savefig(filename, bbox_inches='tight')
        plt.close()# build gif
    with imageio.get_writer(gifFileName, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    # Remove files
    if clearFrames:
        for filename in set(filenames):
            os.remove(filename)

if __name__=="__main__":
    matrixList=None
    
    for d in range(1,2):
        f=f"ionex/codg00{d}0.20i.npy"
        ionex=np.load(f)
        if matrixList is None:
            matrixList=ionex[:24]
        else:
            matrixList=np.concatenate((matrixList,ionex[:24]))
    saveGif(matrixList,'mygif.gif')
            
