import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams["figure.figsize"] = (20,10)
np.random.seed(404)

class LinearReg:

    def __init__(self,alpha=0.005):
        self.alpha=alpha
        

    def dataGen(self,theta=np.random.uniform(0,1,2),samples=500,plot=False):
        temp = dict()
        span=0.5
        
        x=np.linspace(0,10,samples).reshape(samples,1)
        y=theta[0] + theta[1]*x + np.random.uniform(-span,span,[samples,1])
        
        temp['theta']=theta
        temp['x']=x
        temp['y']=y
        self.Data = temp
        
        if plot:
            plt.scatter(x,y)
            plt.grid()
            plt.title('Generate Data')
            plt.show()
        return self

    def JCal(self,i,j):
        
        xData = self.Data['x']
        yData = self.Data['y']
        temp = np.sum(np.square(yData-(i+j*xData)))/(2*len(yData))

        return np.round(temp,4)
        
    def Jfun(self,plot=False):
        
        x     = np.arange(0,1.1,0.05)
        y     = x
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
        ax.scatter(jj,ii,self.JCal(ii,jj), c = 'r', marker='o',s=20,linewidth=2,label='Optimal Point')
        ax.scatter(xx, yy, zz,cmap='viridis', edgecolor='none')

        ax.set_title('Surface plot')
        ax.set_xlabel('theta_1')
        ax.set_ylabel('theta_0')
        ax.set_zlabel('J theta')
        ax.legend()
        
        ax2 = fig.add_subplot(122)
        ax2.scatter(xData,ii+jj*xData,label='actual plot')
        ax2.legend()
        ax2.grid()
        

        for kk in range(2500):

            tHat[0] -= self.alpha * np.sum((tHat[0]+tHat[1]*xData)-yData)/len(xData)
            tHat[1] -= self.alpha * np.sum( ((tHat[0]+tHat[1]*xData)-yData) * xData )/len(xData)

            if not kk%100:
                
                ax.scatter(tHat[1],tHat[0],self.JCal(tHat[0],tHat[1]), c = 'm', marker='o',s=20,linewidth=2)
                ax2.plot(xData,tHat[0]+tHat[1]*xData)
                
                plt.draw()
                plt.pause(0.1)

        
        
lr = LinearReg()
data= lr.dataGen(plot=False)
aa = lr.Jfun()
lr.Parameters()

        
        
