import sys, os
sys.path.append("../")
from feelit import utils

from subprocess import call

emotions = utils.LJ40K
temp_folder = '/home/bs980201/projects/github_repo/LJ40K/exp/temp/tf3k1b08idf2_TSVD300+w2vStanford_rmP_Maxis_series+pattern_mincount3_TSVD300_scale'
output_folder = 'out1'
feature_list_file = '/home/doug919/projects/github_repo/LJ40K/example/finals.json'

for e in emotions:
    model_filename = utils.get_file_name_by_emtion(temp_folder, 'model_'+e, ext='.pkl')
    model_file = os.path.join(temp_folder, model_filename)

    scaler_filename = utils.get_file_name_by_emtion(temp_folder, 'scaler_'+e, ext='.pkl')
    scaler_file = os.path.join(temp_folder, scaler_filename)

    call(['python', 'batchTestModel.py', '-s', scaler_file, '-v', model_file, str(emotions.index(e)), feature_list_file])





