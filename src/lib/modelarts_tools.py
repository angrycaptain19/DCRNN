import os
try:
    import moxing as mox
except:
    print('not use moxing')
else:
    print("use moxing")

modelarts_args=None

def prepare_data_on_modelarts(args):
    """
    将OBS上的数据拷贝到ModelArts中
    """

    if not (args.data_url.startswith('s3://') or args.data_url.startswith('obs://')):
        args.data_local = args.data_url
    else:
        args.data_local = os.path.join(args.local_data_root, 'data')
        if not os.path.exists(args.data_local):
            data_dir = os.path.join(args.local_data_root, 'data')
            mox.file.copy_parallel(args.data_url, data_dir)
        else:
            print('args.data_local: %s is already exist, skip copy' % args.data_local)

    if not (args.train_url.startswith('s3://') or args.train_url.startswith('obs://')):
        args.train_local = args.train_url
    else:
        args.train_local = '.'
        if not os.path.exists(args.train_local):
            os.mkdir(args.train_local)

    global modelarts_args
    modelarts_args=args
    return args


def push_data_back_on_modelarts():
    """
    将ModelArts中train_local的数据拷贝到OBS的train_url上
    """
    global modelarts_args
    args=modelarts_args
    assert modelarts_args != None, "modelarts_args can't be None "
    # if args.data_url.startswith('s3://') or args.data_url.startswith('obs://'):
    #         mox.file.copy_parallel(args.data_local,args.data_url)

    if args.train_url.startswith('s3://') or args.train_url.startswith('obs://'):
        mox.file.copy_parallel(args.train_local,args.train_url)

    print('push back train data success, dir is at ', args.train_url)


