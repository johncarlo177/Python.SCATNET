import matplotlib.pyplot as plt, numpy as np
def main():
    models=['3D-CNN','SpectralFormer','SST','GCN-HSI','SCATNet']; params=[2.5,2.8,2.7,2.2,1.6]
    oa_pc=[97.1,98.3,98.1,97.5,98.9]; oa_ip=[93.8,96.8,96.2,95.4,98.2]; oa_hu=[96.2,97.6,97.3,96.7,98.5]
    avg=list(np.round((np.array(oa_pc)+np.array(oa_ip)+np.array(oa_hu))/3,2))
    plt.figure(figsize=(6,4.5),dpi=150)
    for i,n in enumerate(models):
        plt.scatter(params[i],avg[i],s=80,edgecolor='black',linewidth=0.6,zorder=3); plt.text(params[i]+0.04,avg[i]+0.03,n,fontsize=9)
    plt.plot(params,avg,linestyle='--',linewidth=0.8,zorder=1); plt.xlabel('Parameters (Millions)'); plt.ylabel('Overall Accuracy (%, avg of PC/IP/HU-13)')
    plt.title('Fig. 5. OA vs. Parameter Count Comparison'); plt.grid(True,linewidth=0.4,alpha=0.5); plt.tight_layout(); plt.savefig('../outputs/Fig_5_OA_vs_Parameters.png',dpi=300,bbox_inches='tight')
if __name__ == '__main__': main()
