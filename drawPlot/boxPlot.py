import matplotlib.pyplot as plt
import seaborn as sns

# columns = ['sm_20', 'sm_50', 'sm_100', 'station']

def boxPlot (data, columns):
    fig, axs = plt.subplots(3, figsize =(32, 10))
    # fig.suptitle('Boxplot of soil moisture at different depths for 11 test stations)', fontsize=24)

    plotDF = data.loc[:,columns]
    for i in range(len(columns)-1):
        sns.set(font_scale = 2)
        #plt.ylabel('ylabel', fontsize=18)
        sns.boxplot(x="station", y=columns[i], data=plotDF, palette="Blues", ax=axs[i])

    # plt.savefig('M_boxplot_testStations.png')
    plt.savefig("M_boxplot_testStations.pdf", bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    return plt
