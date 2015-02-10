from subprocess import call

def run(eid):
    
    train_data_root = '/home/doug919/projects/data/MKLv2/2000samples_4/train'
    test_data_root = '/home/doug919/projects/data/MKLv2/2000samples_4/test_8000'
    train_data_tag = '800p800n_Xy'
    test_data_tag = 'Csp.Xy'
    output_prefix = 'Seq%d_E4_8000' % (eid)
    nclass_neg = 39;

    cmd = 'matlab -r "mklv2_exp_4(%d, \'%s\', {\'TFIDF\', \'keyword\', \'image_rgba_gist\', \'image_rgba_phog\'}, \'%s\', \'%s\', \'%s\', \'%s\', %ld, 10);exit;" > log/log_seq_%d' % \
        (eid, output_prefix, train_data_root, test_data_root, train_data_tag, test_data_tag, nclass_neg, eid)
    #cmd = 'matlab -r "mklv2_exp_3(%d, \'%s\', {\'image_rgba_gist\', \'image_rgba_phog\'}, \'%s\', \'%s\', \'%s\', \'%s\', true, %f);exit;" > log/log_thread_%d' % \
    #    (eid, output_prefix, train_data_root, test_data_root, train_data_tag, test_data_tag, nclass_neg, eid)

    print '> run:',cmd
    call(cmd, shell=True)

if __name__ == "__main__":

    eids = range(1, 41) 

    for eid in eids:
        run(eid)

