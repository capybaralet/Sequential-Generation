
import numpy
np = numpy
import numpy.random

import sys
#sys.path.append('../3rdparty/bouncingballs')
import bouncing_balls 


numballs = 3
patchsize = 16
numframes = 15 
numpy_rng  = numpy.random.RandomState(1)



def make_movie_image(movie):
    # takes frames as vectors
    nframes, framesize = movie.shape
    framelen = framesize**.5
    image = np.zeros((framelen, nframes*(framelen+1)))
    maxval = np.max(movie)
    print maxval
    for i in range(nframes):
        image[:, i*(framelen+1)+1:(i+1)*(framelen+1)] = movie[i].reshape((framelen, framelen))
        if i < nframes - 1: # not showing up!
            image[:, (i+1)*(framelen+1)] = maxval*np.ones(framelen)
    return image

def CreateMovie(filename, plotter, numberOfFrames, fps):
    import os, sys
    import matplotlib.pyplot as plt
    for i in range(numberOfFrames):
        plotter(i)
        fname = '_tmp%05d.png'%i
        plt.savefig(fname)
        plt.clf()
    os.system("rm "+filename+".mp4")
    os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png "+filename+".mp4")
    os.system("convert -delay 10 -loop 0 _tmp*.png "+filename+".gif")
    os.system("rm _tmp*.png")


def writemovie(filename, movie, fps=2, *args, **kwargs):
    import pylab
    patchsize = movie.shape[1]
    numframes = movie.shape[0]
    vmin = movie.min()
    vmax = movie.max()
    def plotter(i):
        pylab.imshow(movie[i], interpolation='nearest', cmap=pylab.cm.gray, vmin=vmin, vmax=vmax)
        pylab.axis('off')
    CreateMovie(filename, plotter, numframes, fps)


print 'making data...'

numballs = 1
numframes = 20
patchsize = 28


numcases = 70000
trainmovies = numpy.empty((numcases,numframes,patchsize**2), dtype=numpy.float32)
for i in range(numcases):
    #print i
    trainmovies[i] = bouncing_balls.bounce_vec(patchsize, numballs, numframes)

trainmovies = trainmovies.reshape(-1, numframes, patchsize**2)

np.save('/data/lisatmp2/kruegerd/bouncing_balls/bouncing_ball', trainmovies)
#trainmovies = numpy.load("100000bouncingballmovies16x16.npy") 

#numtest = 10000
#testmovies = numpy.empty((numtest,numframes,patchsize**2), dtype=numpy.float32)
#for i in range(numtest):
    #print i
#    testmovies[i] = bouncing_balls.bounce_vec(patchsize, numballs, numframes)

#testmovies = testmovies.reshape(-1, numframes, patchsize**2)

print '... done'

