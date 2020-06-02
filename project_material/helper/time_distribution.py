import logging
from scipy.stats import gamma
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(5, 5)})

logger = logging.getLogger('Logging for `time_distribution.py`')
logger.setLevel(logging.CRITICAL)

class GammaProbabilityGenerator():
    def __init__(self, shape, scale, loc=0, magnitude_multiplier=1, x=None):
        '''The `magnitude_multiplier` is a custom multiplier to adjust the magnitude of gamma dist(distribution).'''
        self.x = x
        self.shape = shape
        self.scale = scale
        self.loc = loc
        self.magnitude_multiplier = magnitude_multiplier
        self.gamma_dist = gamma(a=self.shape, scale=self.scale, loc=self.loc)

    # PDF
    def get_pdf_prob_by_x(self):
        try:
            return self.gamma_dist.pdf(self.x)*self.magnitude_multiplier
        except:
            logger.warning('WARNING: `x` value not value for `get_pdf_prob_by_x()`.')
            return 0

    def show_pdf_prob_by_x(self):
        return ['{0:.5f}'.format(i) for i in self.get_pdf_prob_by_x()]

    def display_pdf_curve(self):
        gamma_data = gamma.rvs(a=self.shape, scale=self.scale, loc=self.loc,
                               size=10000, random_state=888)/self.magnitude_multiplier
        ax = sns.distplot(gamma_data,
                          kde=True,
                          bins=100,
                          color='skyblue',
                          hist_kws={'linewidth': 15, 'alpha': 1})
        ax.set(xlabel='Gamma PDF Distribution', ylabel='Frequency')
        plt.show()

    def describe_pdf_prob(self):
        ci_95_percent = self.gamma_dist.interval(alpha=0.95)

        return {
            'Median:': '{0:.5f}'.format(self.gamma_dist.median()),
            'Mean:': '{0:.5f}'.format(self.gamma_dist.mean()),
            'Std:': '{0:.5f}'.format(self.gamma_dist.std()),
            '95% CI:': ['{0:.5f}'.format(i) for i in (ci_95_percent)],
            '95% CI range:': '{0:.5f}'.format(ci_95_percent[1]-ci_95_percent[0]),
        }

    # CDF
    def get_cdf_prob_by_x(self):
        try:
            return self.gamma_dist.cdf(self.x)*self.magnitude_multiplier
        except:
            logger.warning('WARNING:`x` value not value for `get_cdf_prob_by_x()`.')
            return 0

    def show_cdf_prob_by_x(self):
        return ['{0:.5f}'.format(i) for i in self.get_cdf_prob_by_x()]

def test():
    obj = GammaProbabilityGenerator(
        shape=1,
        scale=5,
        loc=0,
        magnitude_multiplier=1)
    obj.display_pdf_curve()

    x_values = [0, 0.00001, 0.001, 0.1, 1, 2, 3, 100]
    for x in x_values:
        obj.x = x
        print(obj.get_pdf_prob_by_x())

    print('Info:', obj.describe_pdf_prob())

if __name__=='__main__':
    test()