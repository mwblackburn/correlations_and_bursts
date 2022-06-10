def get_path():
    """Sets the file paths for the AllenSDK data needed for the tutorials in this package."""

    # Insert the file path to your data (e.g. "C:\Users\MickeyMouse\MouseBrainData\manifest.json")
    data_root = (
        "C:/Users/Demogorgon/Documents/College/Marcus/Boston University PhD/Ocker Lab"
    )
    manifest_path = f"{data_root}/AllenSDK_Data/manifest.json"
    save_path = f"{data_root}/correlations_and_bursts/data"
    return (data_root, manifest_path, save_path)
