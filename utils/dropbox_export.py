import dropbox

def upload_to_dropbox(token, local_file_path, dropbox_dest_path):
    dbx = dropbox.Dropbox(token)
    with open(local_file_path, "rb") as f:
        dbx.files_upload(f.read(), dropbox_dest_path, mode=dropbox.files.WriteMode.overwrite)
