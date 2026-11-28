import json, matplotlib, matplotlib.ticker as mt; matplotlib.use('Agg'); import matplotlib.pyplot as plt
def load(f):
    rows=[json.loads(l.split(' ',1)[1]) for l in open(f) if l.startswith('WANDB_SYNC ')]
    ep={r['step']:r['metrics'] for r in rows if not r['final']}
    return [ep[k] for k in sorted(ep)]
G=load('geo_pyr7.txt'); A=load('pyr/cheesewaffler__chile3d-pyr7-cosdir-grid14-s100-e180-seed0-t4x2.txt')
B=load('pyr/alexgvr1027__chile3d-pyr7-cosdir-grid14-s100-e180-seed1-t4x2.txt')
c1,c2,c3='#257de0','#f47230','#f2a600'; ink,ink2,grid='#1f1f1f','#6b6b6b','#e6e6e6'
fig,ax=plt.subplots(figsize=(10,5.6),dpi=200)
for M,col,lab in [(A,c2,'GATr cosdir, seed 0 (final 12.74°)'),(B,c3,'GATr cosdir, seed 1 (final 13.11°)'),(G,c1,'DiT + geobias, s = 100 (final 2.00°)')]:
    ax.plot(range(1,len(M)+1),[m['median_rotation_error'] for m in M],color=col,lw=2 if col==c1 else 1.6,label=lab,zorder=3 if col==c1 else 2)
ax.axhline(90/7,color='#9c9c9c',lw=1,ls='--'); ax.text(3,90/7*0.9,'guessing the turn: 90/7 = 12.9°',color=ink2,fontsize=8.5)
ax.plot([183],[2.00],marker='o',ms=8,color=c1,markeredgecolor='white',markeredgewidth=1.5,clip_on=False)
ax.annotate('final, 32-sample medoid: 2.00° (acc@15 0.982)',(181,2.0),xytext=(120,2.0),textcoords='data',ha='right',va='center',fontsize=8.5,color=ink,arrowprops=dict(arrowstyle='->',color='#9c9c9c',lw=1))
ax.set_yscale('log'); ax.yaxis.set_minor_locator(mt.NullLocator()); ax.yaxis.set_minor_formatter(mt.NullFormatter())
t=[2,3,5,10,20,50,100]; ax.set_yticks(t); ax.set_yticklabels([str(v) for v in t]); ax.set_xlim(0,188); ax.set_ylim(1.7,100)
ax.grid(True,axis='y',color=grid,lw=.8)
for s in ['top','right']: ax.spines[s].set_visible(False)
for s in ['left','bottom']: ax.spines[s].set_color('#bdbdbd')
ax.tick_params(colors=ink2); ax.set_xlabel('epoch',color=ink2); ax.set_ylabel('median rotation error, degrees (log)',color=ink2)
ax.set_title('Full-3D SYMSOL-CHILE 7-gon (pyr7): same recipe, two vector fields',loc='left',color=ink,fontsize=12,fontweight='bold',pad=24)
ax.text(0,1.025,'14×14 map, 368 sphere tokens, constant LR 1e-4, 4,500 views, 1,000 test images. Lines: 1 sample per image after each epoch.',transform=ax.transAxes,color=ink2,fontsize=8.5)
ax.legend(frameon=False,fontsize=9,loc='upper right',labelcolor=ink)
fig.savefig('/home/user/GA-Research/pyr7_geobias_vs_gatr.png',bbox_inches='tight',facecolor='white')
