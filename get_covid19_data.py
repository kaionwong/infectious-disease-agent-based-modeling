import git
import os

def main(save_switch=False):
    if save_switch:
        git.Git(os.getcwd()+'\\data').clone('https://github.com/ishaberry/Covid19Canada.git')

if __name__ == '__main__':
    main(save_switch=True)