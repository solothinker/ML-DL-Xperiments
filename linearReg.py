import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,10)
np.random.seed(404)

class LinearReg:

    def __init__(self,alpha=0.0001):
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
                temp.append(self.JCal(jj,ii))
            z.append(temp)
        zz=np.matrix(z)
##        if plot:
##            fig = plt.figure()
##            ax = plt.axes(projection='3d')
##            ii,jj=self.Data['theta'][0],self.Data['theta'][1]
##
##            ax.scatter(ii,jj,self.JCal(xData,yData,jj,ii), c = 'r', marker='o',s=20,linewidth=2,label='Optimal Point')
##            ax.scatter(xx, yy, zz,cmap='viridis', edgecolor='none')
##
##            ax.set_title('Surface plot')
##            ax.set_xlabel('xx')
##            ax.set_ylabel('yy')
##            ax.set_zlabel('J theta')
##
##            plt.legend()
##            plt.show()
            
        JSurf['xx']=xx
        JSurf['yy']=yy
        JSurf['ZZ']=zz
        self.JSurf = JSurf
        return self
    
    def Parameters(self):
        
        tHat = np.random.uniform(0,1,2)

        xData = self.Data['x']
        yData = self.Data['y']
        
        xx = self.JSurf['xx']
        yy = self.JSurf['yy']
        zz = self.JSurf['ZZ']
        
        ii,jj=self.Data['theta'][0],self.Data['theta'][1]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(jj,ii,self.JCal(ii,jj), c = 'r', marker='o',s=20,linewidth=2,label='Optimal Point')
        for kk in range(1000):
            #,label='Tuning Point')
            tHat[0] -= self.alpha * np.sum(yData - (tHat[0]+tHat[1]*xData))/len(xData)
            tHat[1] -= self.alpha * np.sum( (yData - (tHat[0]+tHat[1]*xData)) * xData )/len(xData)
##            print(self.JCal(tHat[1],tHat[0]))
##            if not kk%10:
##                ax.scatter(tHat[0],tHat[1],self.JCal(tHat[1],tHat[0]), c = 'm', marker='o',s=20,linewidth=2)
        
        ax.scatter(xx, yy, zz,cmap='viridis', edgecolor='none')
        ax.set_title('Surface plot')
        ax.set_xlabel('xx')
        ax.set_ylabel('yy')
        ax.set_zlabel('J theta')

        plt.legend()
        plt.show()
        
        
        
lr = LinearReg()
data= lr.dataGen(plot=False)
aa = lr.Jfun()
lr.Parameters()
##print(data)
        
        
