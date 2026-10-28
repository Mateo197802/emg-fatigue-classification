import pandas as pd


def test_public_results_exclude_identity_paths_and_parameters(tmp_path):
    from emg_fatigue.reporting import export_public_results
    frame=pd.DataFrame([{'model':'majority','fold_id':'F01','subject_id':'private-name','source_member':'person/secret.csv','parameters':'private','accuracy':.5,'f1_macro':.4,'n_test':10}])
    p=export_public_results(frame,tmp_path/'public',tmp_path/'private')
    text=p.read_text()
    assert 'private-name' not in text and 'secret.csv' not in text and 'parameters' not in text
    assert 'F01' not in text and 'accuracy' in text and 'f1_macro' in text
    assert p.read_bytes()==export_public_results(frame,tmp_path/'public',tmp_path/'private').read_bytes()
