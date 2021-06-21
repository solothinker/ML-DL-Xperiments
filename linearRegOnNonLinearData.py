import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

plt.rcParams["figure.figsize"] = (20,10)
np.random.seed(404)

class LinearReg:
    '''implementing y=a1*x+a2*x^2'''
    def __init__(self,alpha=0.0002):
        self.alpha=alpha
        

    def dataGen(self,theta=np.random.uniform(0,1,2),samples=500,plot=False):
        temp = dict()
        span=0.5
        
        x=np.linspace(0,10,samples).reshape(samples,1)
        y=theta[0]*x + theta[1]*x**2 + np.random.uniform(-span,span,[samples,1])
        
        temp['theta']=theta
        temp['x']=x
        temp['y']=y
        self.Data = temp
        
        if plot:
            ax = plt.axes(projection='3d')
            ax.scatter3D(x,x**2,y)
            plt.grid()
            plt.title('Generate Data')
            plt.show()
        return self

    def JCal(self,i,j):
        
        xData = self.Data['x']
        yData = self.Data['y']
        temp = np.sum(np.square(yData-(i * xData +j * xData**2 )))/(2*len(yData))

        return np.round(temp,4)
        
    def Jfun(self,plot=False):
        
        x     = np.arange(-1 ,1.1,0.05)
        y     = x**2
        z     = []
        JSurf = dict()
        xx,yy = np.meshgrid(x,y)#,sparse=True)
        
        for ii in x:
            temp = []
            for jj in y:
                temp.append(self.JCal(ii,jj))
            z.append(temp)
        zz=np.matrix(z)
            
        JSurf['xx']=xx
        JSurf['yy']=yy
        JSurf['ZZ']=zz
        self.JSurf = JSurf
        return self
    
    def Parameters(self):
        
        tHat = np.random.uniform(0,0.1,2)

        xData = self.Data['x']
        yData = self.Data['y']
        
        xx = self.JSurf['xx']
        yy = self.JSurf['yy']
        zz = self.JSurf['ZZ']

        ii,jj=self.Data['theta'][0],self.Data['theta'][1]
        fig = plt.figure()

        ax = fig.add_subplot(121,projection='3d')
        ax.scatter(ii,jj,self.JCal(ii,jj), c = 'r', marker='o',s=20,linewidth=2,label='Optimal Point')
        ax.scatter(xx, yy, zz,cmap='viridis', edgecolor='none')

        ax.set_title('Surface plot')
        ax.set_xlabel('theta_1')
        ax.set_ylabel('theta_0')
        ax.set_zlabel('J theta')
        ax.legend()
        
        ax2 = fig.add_subplot(122,projection='3d')
        ax2.scatter(xData , xData**2, ii * xData + jj * xData**2,label='actual plot')
        ax2.legend()
        ax2.grid()
        
        filenames = []
        for kk in range(6000):

            tHat[0] -= self.alpha * np.sum( ((tHat[0] * xData + tHat[1] * xData**2)-yData) * xData ** 1)/len(xData)
            tHat[1] -= self.alpha * np.sum( ((tHat[0] * xData + tHat[1] * xData**2)-yData) * xData **2 )/len(xData)

            if not kk%200:

                ax.scatter(tHat[0],tHat[1],self.JCal(tHat[0],tHat[1]), c = 'm', marker='o',s=20,linewidth=2)
                ax2.scatter(xData,xData**2,tHat[0]*xData+tHat[1]*xData**2)
##                filename = f'{kk}.png'
##                filenames.append(filename)
                plt.draw()
                plt.pause(0.1)
##                plt.savefig(filename)

##        with imageio.get_writer('nongif.gif', mode='I') as writer:
##            for filename in filenames:
##                image = imageio.imread(filename)
##                writer.append_data(image)
##        
##        # Remove files
##        for filename in set(filenames):
##            os.remove(filename)


        
lr = LinearReg()
data= lr.dataGen(plot=False)
aa = lr.Jfun()
lr.Parameters()

        
        
