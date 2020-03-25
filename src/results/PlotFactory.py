import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', size=16)
plt.rc('axes', titlesize=26)
plt.rcParams["figure.figsize"] = [16,10]

class PlotFactory():

    def create_scatter(self, x, y, xlab, ylab, title):
        plt.scatter(x,y)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        max_point = min(max(list(x)),max(list(y)))
        plt.plot([0,max_point],[0,max_point],'--') # identity line
        plt.show()

    def create_heatmap(self, x, y, xlab, ylab, title):
        plt.hist2d(x,y, bins=100)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.ylim(0,10000)
        plt.xlim(0,10000)
        plt.title(title)
        max_point = min(max(list(x)),max(list(y)))
        plt.plot([0,10000],[0,10000],'--') # identity line
        plt.show()


    def perc_corr(self, y_test, pred, x):
        error = abs(y_test-pred)
        y = [sum(error/y_test < i)/len(error) for i in x]
        return(y)

    def perc_perc_corr(self, y_test, pred, x):
        error = abs(y_test-pred)
        y = [sum(error < i)/len(error) for i in x]
        return(y)

    def plot_curve(self, y_test, pred_dict, step, xlab, ylab, title):
        x = [x/step for x in range(step+1)]
        corr_dict = {pred:self.perc_corr(y_test, pred_dict[pred], x) for pred in pred_dict}

        plt.figure(figsize=(12,8))

        for corr in corr_dict:
            plt.plot(x,corr_dict[corr], label=corr)

        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_perc_curve(self, y_test, pred_dict, step, xlab, ylab, title):
        x = [x/step for x in range(step+1)]
        corr_dict = {pred:self.perc_perc_corr(y_test, pred_dict[pred], x) for pred in pred_dict}

        plt.figure(figsize=(12,8))

        for corr in corr_dict:
            plt.plot(x,corr_dict[corr], label=corr)

        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()
