def get_pos(fname):
    """Get the local position of an udacity log file."""
    
    data = pd.read_table(fname, names=['raw_data'])
    data['command'] = data.raw_data.apply(lambda x: x.split(',')[0])

    pos = data[data.command == 'MsgID.LOCAL_POSITION']
    pos = pos.raw_data.str.split(',', expand=True).drop(columns=0)
    pos.columns = ['t', 'x', 'y', 'z']
    pos = pos.astype(np.float)
    
    return pos
