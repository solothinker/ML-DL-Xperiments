import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,10)
np.random.seed(404)

class LinearReg:

    def __init__(self,alpha=0.001):
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

    def Jfun(self):
        
        x     = np.arange(0,1.1,0.05)
        y     = x
        z     = []
        
        xData = self.Data['x']
        yData = self.Data['y']
        xx,yy = np.meshgrid(x,y)#,sparse=True)
        
        for ii in x:
            temp = []
            for jj in y:
                temp.append(np.sum(np.square(yData-(jj+ii*xData)))/(2*len(yData)))
            z.append(temp)

        zz=np.matrix(z)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ii,jj=self.Data['theta'][0],self.Data['theta'][1]
        ax.scatter(ii,jj,np.sum(np.square(yData-(ii+jj*xData)))/(2*len(yData)), c = 'r', marker='o',s=20,linewidth=2,label='Optimal Point')
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
##print(data)
        
        
