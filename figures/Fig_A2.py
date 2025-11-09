import numpy as np, matplotlib.pyplot as plt
def main():
    np.random.seed(7); heads,bands=8,50; attn=np.random.rand(heads,bands)
    for h in range(heads): attn[h]=np.convolve(attn[h],np.ones(5)/5,mode='same')
    fig,ax=plt.subplots(figsize=(7,4.2),dpi=150); im=ax.imshow(attn,aspect='auto',cmap='plasma'); cbar=plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    cbar.set_label('Attention Weight'); ax.set_xlabel('Spectral Band Index'); ax.set_ylabel('Attention Head'); ax.set_title('Fig. A2. Feature Attention Map Visualization')
    plt.tight_layout(); plt.savefig('../outputs/Fig_A2_AttentionMap.png',dpi=300,bbox_inches='tight')
if __name__ == '__main__': main()
