import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium", app_title="Tornado Viewer")


@app.cell
def import_marimo():
    # Load the marimo library in a dedicated cell for efficiency
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Fred Hutch Bioinformatics Core: Tornado Viewer""")
    return


@app.cell
def cirro_dataset_type_filter():
    # Define the types of datasets which can be read in
    # This is used to filter the dataset selector, below
    cirro_dataset_type_filter = ["custom_dataset"]
    return (cirro_dataset_type_filter,)


@app.cell
def loading_dependencies(mo):
    with mo.status.spinner("Loading dependencies"):
        # If we are running in WASM, some dependencies need to be set up appropriately.
        # This is really just aligning the needs of the app with the default library versions
        # that come when a marimo app loads in WASM.
        from io import StringIO, BytesIO
        from typing import Dict, List
        from matplotlib import pyplot as plt
        from matplotlib.axes._axes import Axes
        import matplotlib.colors as mcolors
        import pandas as pd
        import numpy as np
        from functools import lru_cache
        import pyBigWig
        import tempfile
        from scipy import stats

        from cirro import DataPortalLogin
        from cirro.config import list_tenants
    return (
        Axes,
        BytesIO,
        DataPortalLogin,
        Dict,
        List,
        StringIO,
        list_tenants,
        lru_cache,
        np,
        pd,
        plt,
        pyBigWig,
        stats,
        tempfile,
    )


@app.cell
def _(mo):
    # Get and set the query parameters
    query_params = mo.query_params()
    return (query_params,)


@app.cell
def def_domain_to_name(list_tenants):
    # Get the tenants (organizations) available in Cirro
    tenants_by_name = {i["displayName"]: i for i in list_tenants()}
    tenants_by_domain = {i["domain"]: i for i in list_tenants()}


    def domain_to_name(domain):
        return tenants_by_domain.get(domain, {}).get("displayName")


    def name_to_domain(name):
        return tenants_by_name.get(name, {}).get("domain")
    return domain_to_name, tenants_by_name


@app.cell
def _(mo):
    mo.md(r"""## Load Data""")
    return


@app.cell
def _(mo):
    # Use a state element to manage the Cirro client object
    get_client, set_client = mo.state(None)
    return get_client, set_client


@app.cell
def domain_ui(domain_to_name, mo, query_params, tenants_by_name):
    # Let the user select which tenant to log in to (using displayName)
    domain_ui = mo.ui.dropdown(
        options=tenants_by_name,
        value=domain_to_name(query_params.get("domain")),
        on_change=lambda i: query_params.set("domain", i["domain"]),
        label="Load Data from Cirro",
    )
    domain_ui
    return (domain_ui,)


@app.cell
def cirro_login_ui(DataPortalLogin, domain_ui, get_client, mo):
    # If the user is not yet logged in, and a domain is selected, then give the user instructions for logging in
    # The configuration of this cell and the two below it serve the function of:
    #   1. Showing the user the login instructions if they have selected a Cirro domain
    #   2. Removing the login instructions as soon as they have completed the login flow
    if get_client() is None and domain_ui.value is not None:
        with mo.status.spinner("Authenticating"):
            # Use device code authorization to log in to Cirro
            cirro_login = DataPortalLogin(base_url=domain_ui.value["domain"])
            cirro_login_ui = mo.md(cirro_login.auth_message_markdown)
    else:
        cirro_login = None
        cirro_login_ui = None

    mo.stop(cirro_login is None)
    cirro_login_ui
    return (cirro_login,)


@app.cell
def set_client(cirro_login, set_client):
    # Once the user logs in, set the state for the client object
    set_client(cirro_login.await_completion())
    return


@app.cell
def get_client(get_client, mo):
    # Get the Cirro client object (but only take action if the user selected Cirro as the input)
    client = get_client()
    mo.stop(client is None)
    return (client,)


@app.cell
def def_id_to_name():
    # Helper functions for dealing with lists of objects that may be accessed by id or name
    def id_to_name(obj_list: list, id: str) -> str:
        if obj_list is not None:
            return {i.id: i.name for i in obj_list}.get(id)


    def name_to_id(obj_list: list) -> dict:
        if obj_list is not None:
            return {i.name: i.id for i in obj_list}
        else:
            return {}
    return id_to_name, name_to_id


@app.cell
def list_projects(client):
    # Set the list of projects available to the user
    projects = client.list_projects()
    projects.sort(key=lambda i: i.name)
    return (projects,)


@app.cell
def project_ui(id_to_name, mo, name_to_id, projects, query_params):
    # Let the user select which project to get data from
    project_ui = mo.ui.dropdown(
        label="Project:",
        value=id_to_name(projects, query_params.get("project")),
        options=name_to_id(projects),
        on_change=lambda i: query_params.set("project", i)
    )
    project_ui
    return (project_ui,)


@app.cell
def list_datasets(cirro_dataset_type_filter, client, mo, project_ui):
    # Stop if the user has not selected a project
    mo.stop(project_ui.value is None)

    # Get the list of datasets available to the user
    # Filter the list of datasets by type (process_id)
    datasets = [
        dataset
        for dataset in client.get_project_by_id(project_ui.value).list_datasets()
        if dataset.process_id in cirro_dataset_type_filter
    ]
    datasets.sort(key=lambda d: d.created_at, reverse=True)
    return (datasets,)


@app.cell
def dataset_ui(datasets, id_to_name, mo, name_to_id, query_params):
    # Let the user select which dataset to get data from
    dataset_ui = mo.ui.dropdown(
        label="Dataset:",
        value=id_to_name(datasets, query_params.get("dataset")),
        options=name_to_id(datasets),
        on_change=lambda i: query_params.set("dataset", i)
    )
    dataset_ui
    return (dataset_ui,)


@app.cell
def def_list_files(client, lru_cache):
    # Cached functions for reading from a Cirro dataset
    @lru_cache
    def list_files(project_id: str, dataset_id: str):
        return [
            file.name
            for file in (
                client
                .get_project_by_id(project_id)
                .get_dataset_by_id(dataset_id)
                .list_files()
            )
        ]


    @lru_cache
    def read_file(project_id: str, dataset_id: str, file_path: str, **kwargs):
        return (
            client
            .get_project_by_id(project_id)
            .get_dataset_by_id(dataset_id)
            .list_files()
            .get_by_id(file_path)
            .read(**kwargs)
        )
    return list_files, read_file


@app.cell
def list_files(dataset_ui, list_files, mo, project_ui):
    # Stop if the user has not selected a dataset
    mo.stop(dataset_ui.value is None)

    # Get the list of all files in the selected dataset
    all_files = list_files(project_ui.value, dataset_ui.value)
    return (all_files,)


@app.cell
def def_filter_files(all_files):
    def filter_files(prefix=None, suffix=None, contains=None):
        return [
            fn for fn in all_files
            if
            (prefix is None or fn.startswith(prefix) or fn.startswith("data/" + prefix))
            and
            (suffix is None or fn.endswith(suffix) or fn.endswith("data/" + suffix))
            and
            (contains is None or contains in fn)
        ]
    return (filter_files,)


@app.cell
def _(mo):
    mo.md(r"""### Select Regions (BED)""")
    return


@app.cell
def select_bed(filter_files, mo):
    # Ask the user to select a BED file
    bed_files = filter_files(suffix=".txt")
    select_bed = mo.ui.dropdown(
        label="Select BED file:",
        options=bed_files if len(bed_files) > 0 else ["No BED Files Found"],
        value=bed_files[-1] if len(bed_files) > 0 else "No BED Files Found"
    )
    select_bed
    return (select_bed,)


@app.cell
def read_bed_file(StringIO, lru_cache, mo, pd, read_file):
    # Read in the BED file
    @lru_cache
    def read_bed(project: str, dataset: str, file: str):
        with mo.status.spinner("Reading BED file..."):
            return pd.read_csv(
                StringIO(read_file(project, dataset, file)),
                sep="\t",
                # header=None
            # ).rename(
            #     columns=dict(zip(range(5), ['chr', 'start', 'end', 'id', 'peak_group']))
            ).fillna("None")

    return (read_bed,)


@app.cell
def plot_peak_groups():
    # # Show the number of different peak groups
    # mo.stop(select_bed.value == "No BED Files Found")
    # read_bed(project_ui.value, dataset_ui.value, select_bed.value)["peak_group"].value_counts().plot(kind="bar")
    # plt.ylabel("Number of Peaks")
    # plt.xlabel("Peak Group")
    # plt.title("Number of Peaks by Group")
    return


@app.cell
def _(mo):
    mo.md(r"""### Select Genome Coverage Tracks""")
    return


@app.cell
def select_bigwigs(filter_files, mo):
    # Ask the user to select one or more bigWig files
    bigwig_file_options = filter_files(suffix=".bigWig")
    select_bigWigs = mo.ui.multiselect(
        label="Select bigWig file(s):",
        options=bigwig_file_options if len(bigwig_file_options) > 0 else ["No bigWig Files Found"],
        value=[] if len(bigwig_file_options) > 0 else ["No bigWig Files Found"]
    )
    select_bigWigs
    return (select_bigWigs,)


@app.cell
def _(mo, select_bigWigs):
    if len(select_bigWigs.value) == 0:
        _out = mo.md("Please select bigWig files for analysis")
    else:
        _out = None
    _out

    return


@app.cell
def _(mo):
    # Cache params in the state
    get_size, set_size = mo.state(2000)
    get_n_bins, set_n_bins = mo.state(200)
    get_ref, set_ref = mo.state("Middle")
    get_justification, set_justification = mo.state("Center")
    return (
        get_justification,
        get_n_bins,
        get_ref,
        get_size,
        set_justification,
        set_n_bins,
        set_ref,
        set_size,
    )


@app.cell
def window_ui(
    get_justification,
    get_n_bins,
    get_ref,
    get_size,
    mo,
    select_bigWigs,
    set_justification,
    set_n_bins,
    set_ref,
    set_size,
):
    # Get options for how the windows will be set up
    mo.stop(len(select_bigWigs.value) == 0 or select_bigWigs.value == ["No bigWig Files Found"])
    window_ui = mo.md("""
    ### Set Window Size / Position

    - {size}
    - {n_bins}
    - {ref}
    - {justification}
    """).batch(
        size=mo.ui.number(label="Window Size:", value=get_size(), on_change=set_size),
        n_bins=mo.ui.number(label="Number of Bins:", value=get_n_bins(), on_change=set_n_bins),
        ref=mo.ui.dropdown(label="Region Reference Point:", options=["Start", "Middle", "End"], value=get_ref(), on_change=set_ref),
        justification=mo.ui.dropdown(label="Window Justification:", options=["Left", "Center", "Right"], value=get_justification(), on_change=set_justification),
    )
    window_ui
    return (window_ui,)


@app.cell
def make_windows(
    dataset_ui,
    lru_cache,
    np,
    project_ui,
    read_bed,
    select_bed,
    window_ui,
):
    # Set up a table with the actual window coordinates
    @lru_cache
    def _make_windows(
        project: str,
        dataset: str,
        file: str,
        size: int,
        ref: str,
        justification: str
    ):
        bed = read_bed(project, dataset, file)
        if ref == "Start":
            ref = bed['start']
        elif ref == "Middle":
            ref = bed[['start', 'end']].apply(np.mean, axis=1)
        elif ref == "End":
            ref = bed['end']
        else:
            raise ValueError(f"Did not expect ref == '{ref}'")

        if justification == "Left":
            start, end = ref, ref + size
        elif justification == "Center":
            start, end = ref - (size / 2.), ref + (size / 2)
        elif justification == "Right":
            start, end = ref - size, ref
        else:
            raise ValueError(f"Did not expect justification == '{justification}'")

        return bed.assign(
            window_start=start.apply(int),
            window_end=end.apply(int)
        ).query(
            "window_start >= 0"
        )

    windows = _make_windows(
        project_ui.value,
        dataset_ui.value,
        select_bed.value,
        window_ui.value['size'],
        window_ui.value['ref'],
        window_ui.value['justification']
    )
    return (windows,)


@app.cell
def def_download_bigwig(client, lru_cache, pyBigWig, tempfile):
    @lru_cache
    def download_bigWig(project_id: str, dataset_id: str, fn: str):
        with tempfile.TemporaryDirectory() as tmp:
            (
                client
                .get_dataset(project_id, dataset_id)
                .list_files()
                .get_by_id(fn)
                .download(download_location=tmp)
            )
            return pyBigWig.open(f"{tmp}/{fn}")

    return (download_bigWig,)


@app.cell
def def_read_bigwig(download_bigWig, lru_cache, mo):
    # Read in the select_bigWigs files
    @lru_cache
    def read_bigWig(project_id: str, dataset_id: str, fn: str):
        with mo.status.spinner(f"Reading {fn}..."):
            return download_bigWig(project_id, dataset_id, fn)

    return (read_bigWig,)


@app.cell
def get_windows(
    Dict,
    dataset_ui,
    lru_cache,
    mo,
    np,
    pd,
    project_ui,
    read_bigWig,
    stats,
    windows,
):
    # Compute the windows
    def _get_window(wig: Dict[str, np.array], r: pd.Series, sub_bar: mo.status.progress_bar, n_bins: int):
        sub_bar.update()
        try:
            return stats.binned_statistic(
                range(r['window_start'], r['window_end']),
                np.nan_to_num(wig.values(r['chrom'], r['window_start'], r['window_end']), nan=0.0),
                'mean',
                bins=n_bins
            ).statistic
        except Exception as e:
            print(r)
            raise e


    @lru_cache
    def get_windows(kw: str, n_bins: int):

        # Read the wig
        wig = read_bigWig(project_ui.value, dataset_ui.value, kw)

        # Get the chromosome lengths
        chrlens = wig.chroms()

        # Filter down to the valid windows which are contained within the available chromosomes
        _windows = windows.loc[
            windows.apply(
                lambda r: chrlens.get(r['chrom'], 0) >= r['window_end'] and r['window_start'] > 0,
                axis=1
            )
        ]

        with mo.status.progress_bar(
            title=kw,
            total=_windows.shape[0],
            remove_on_exit=True
        ) as sub_bar:
            return pd.DataFrame([
                _get_window(wig, r, sub_bar, n_bins)
                for _, r in _windows.iterrows()
            ], index=_windows.index).fillna(0).astype(float)

    return (get_windows,)


@app.cell
def _(windows):
    # Get the options of columns names to use for splitting peaks by (optionally)
    peak_group_cname_options = [
        cname for cname in windows.columns.values
        if cname not in ["chrom", "start", "end", "peak_ID", "window_start", "window_end"]
    ]
    return (peak_group_cname_options,)


@app.cell
def _(mo):
    get_cnames_filter_windows, set_cnames_filter_windows = mo.state([])
    return


@app.cell
def _(get_windows, mo, select_bigWigs, window_ui):
    # Populate the dict with each of the selected samples
    window_dfs = {}

    with mo.status.progress_bar(
        title="Calculating window coverage...",
        total=len(select_bigWigs.value),
        remove_on_exit=True
    ) as bar:

        # Iterate over each selected input file
        for fn in select_bigWigs.value:
            # Get the windows
            window_dfs[fn] = get_windows(fn, window_ui.value["n_bins"])
            bar.update()
    return (window_dfs,)


@app.cell
def _(mo, peak_group_cname_options, windows):
    # Let the user filter which windows are used for plotting
    mo.stop(windows is None)
    filter_windows_ui = mo.md("""
    ### Filter Windows

    """ + "\n".join([
        "- {" + cname + "}"
        for cname in peak_group_cname_options
    ])).batch(
        **{
            cname: mo.ui.multiselect(
                label=cname,
                options=windows[cname].dropna().value_counts().index.values,
                value=windows[cname].unique()
            )
            for cname in peak_group_cname_options
        }
    )

    filter_windows_ui
    return (filter_windows_ui,)


@app.cell
def _(filter_windows_ui, mo, pd, peak_group_cname_options, windows):
    # Apply the filtering selected by the user
    def _passes_filter(r: pd.Series) -> bool:
        for cname in peak_group_cname_options:
            if r[cname] not in filter_windows_ui.value[cname]:
                return False
        return True


    filtered_windows = (
        windows
        .assign(passes_filter=windows.apply(_passes_filter, axis=1))
        .query("passes_filter")
        .drop(columns=["passes_filter"])
    )
    mo.md(f"Number of windows passing filter: {filtered_windows.shape[0]:,}")
    return (filtered_windows,)


@app.cell
def _(filtered_windows, window_dfs):
    filtered_window_dfs = {kw: val.reindex(index=filtered_windows.index) for kw, val in window_dfs.items()}
    return (filtered_window_dfs,)


@app.cell
def sample_annot_ui(filtered_windows, mo, select_bigWigs):
    # Let the user rename and group the wig files
    mo.stop(filtered_windows is None)
    sample_annot_ui = mo.md(
        """### Dataset Names

    Datasets annotated with the same name will be averaged together.

        """ +
        '\n'.join([
            '\n'.join([
                '{' + kw + '_' + str(sample_ix) + '}'
                for kw in ['name']
            ])
            for sample_ix, sample_name in enumerate(select_bigWigs.value)
        ])
    ).batch(**{
        f"name_{sample_ix}": mo.ui.text(
            label=sample_name,
            value=sample_name.split("/")[-1][:-len(".bigWig")],
            full_width=True
        )
        for sample_ix, sample_name in enumerate(select_bigWigs.value)
    })
    sample_annot_ui
    return (sample_annot_ui,)


@app.cell
def _(filtered_window_dfs, mo, sample_annot_ui):
    # Apply labels for each selected wig file and merge replicates
    def _merge_window_data(filtered_window_dfs, wig_labels):
        # Make a list of the labels that were applied by the user
        labels = [wig_labels[f"name_{ix}"] for ix in range(len(filtered_window_dfs))]

        merged = {}
        for label in labels:
            if label in merged:
                continue
            kws = [kw for kw, _label in zip(filtered_window_dfs.keys(), labels) if _label == label]
            if len(kws) == 1:
                merged[label] = filtered_window_dfs[kws[0]]
            else:
                merged[label] = sum([filtered_window_dfs[kw] for kw in kws]) / len(kws)

        return merged

    with mo.status.spinner("Merging replicates...", remove_on_exit=True):
        data = _merge_window_data(filtered_window_dfs, sample_annot_ui.value)

    return (data,)


@app.cell
def _(max_val, mo):
    get_cname_peak_groups, set_cname_peak_groups = mo.state("None")
    get_max_val, set_max_val = mo.state(max_val)
    get_heatmap_height, set_heatmap_height = mo.state(0.75)
    get_figure_height, set_figure_height = mo.state(6)
    get_panel_width, set_panel_width = mo.state(2)
    get_title_size, set_title_size = mo.state(8)
    return (
        get_cname_peak_groups,
        get_figure_height,
        get_heatmap_height,
        get_max_val,
        get_panel_width,
        get_title_size,
        set_cname_peak_groups,
        set_figure_height,
        set_heatmap_height,
        set_max_val,
        set_panel_width,
        set_title_size,
    )


@app.cell
def _(
    filtered_window_dfs,
    get_cname_peak_groups,
    get_figure_height,
    get_heatmap_height,
    get_max_val,
    get_panel_width,
    get_title_size,
    mo,
    peak_group_cname_options,
    set_cname_peak_groups,
    set_figure_height,
    set_heatmap_height,
    set_max_val,
    set_panel_width,
    set_title_size,
):
    mo.stop(len(filtered_window_dfs) == 0)

    params = mo.md("""
    ### Plot Settings

    - {max_val}
    - {heatmap_height}
    - {cmap}
    - {figure_height}
    - {panel_width},
    - {title_size}
    - {cname_peak_groups}
    """).batch(
        cname_peak_groups=mo.ui.dropdown(
            label="Split Peaks By:",
            options=["None"] + peak_group_cname_options,
            value=get_cname_peak_groups(),
            on_change=set_cname_peak_groups
        ),
        max_val=mo.ui.number(
            label="Maximum Value (Heatmap):",
            value=get_max_val(),
            on_change=set_max_val
        ),
        heatmap_height=mo.ui.number(
            label="Heatmap Height Ratio:",
            start=0.1,
            stop=0.9,
            value=get_heatmap_height(),
            on_change=set_heatmap_height
        ),
        figure_height=mo.ui.number(
            label="Figure Height:",
            start=1,
            stop=100,
            step=1,
            value=get_figure_height(),
            on_change=set_figure_height
        ),
        panel_width=mo.ui.number(
            label="Panel Width:",
            start=1,
            stop=100,
            step=1,
            value=get_panel_width(),
            on_change=set_panel_width
        ),
        title_size=mo.ui.number(
            label="Panel Font Size:",
            start=1,
            stop=100,
            step=1,
            value=get_title_size(),
            on_change=set_title_size
        ),
        cmap=mo.ui.dropdown(
            label="Color Pallete",
            options=['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                      'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
                      'berlin', 'managua', 'vanimo'],
            value="bwr"
        )
    )
    params
    return (params,)


@app.cell
def _(filtered_windows, mo, params):
    get_peak_groups, set_peak_groups = mo.state((
        filtered_windows[params.value["cname_peak_groups"]].value_counts().index.values[:1]
        if params.value["cname_peak_groups"] != "None"
        else []
    ))
    return get_peak_groups, set_peak_groups


@app.cell
def _(filtered_windows, get_peak_groups, mo, params, set_peak_groups):
    # Select peak groups to display
    if params.value["cname_peak_groups"] != "None":
        select_peaks = mo.md("{groups}").batch(
            groups=mo.ui.multiselect(
                label="Peak Groups:",
                options=filtered_windows[params.value["cname_peak_groups"]].value_counts().index.values,
                value=get_peak_groups(),
                on_change=set_peak_groups
            )
        )
    else:
        select_peaks = mo.md("").batch()
    select_peaks
    return (select_peaks,)


@app.cell
def _(mo, select_peaks):
    rename_peaks_ui = mo.md("\n".join([
        "- {" + peak_group + "}"
        for peak_group in select_peaks.value.get("groups", [])
    ])).batch(
        **{
            peak_group: mo.ui.text(
                label=f"{peak_group}: ",
                value=peak_group
            )
            for peak_group in select_peaks.value.get("groups", [])
        }
    )
    rename_peaks_ui
    return (rename_peaks_ui,)


@app.cell
def _(data):
    # Show the distribution of all values
    max_val = max(*[df.quantile(0.9).max() for df in data.values()])
    min_val = min(*[df.min().min() for df in data.values()])
    return (max_val,)


@app.cell
def _():
    # # Let the user select the point on the colormap scale for each of three colors
    # color_options = [n.replace("tab:", "") for n in mcolors.TABLEAU_COLORS]
    # cmap_ui = mo.md("""
    # - {color_1} {value_1}
    # - {color_2} {value_2}
    # - {color_3} {value_3}
    # """).batch(
    #     color_1=mo.ui.dropdown(label="Color 1:", value="blue", options=color_options),
    #     value_1=mo.ui.slider(value=min_val, start=min_val, stop=max_val, show_value=True),
    #     color_2=mo.ui.dropdown(label="Color 2:", value="orange", options=color_options),
    #     value_2=mo.ui.slider(value=np.mean([min_val, max_val]), start=min_val, stop=max_val, show_value=True),
    #     color_3=mo.ui.dropdown(label="Color 3:", value="red", options=color_options),
    #     value_3=mo.ui.slider(value=max_val, start=min_val, stop=max_val, show_value=True)
    # )
    # cmap_ui
    return


@app.cell
def _(
    Axes,
    BytesIO,
    Dict,
    List,
    data,
    filtered_windows,
    params,
    pd,
    plt,
    rename_peaks_ui,
    select_peaks,
    window_ui,
):
    def plot_data(
        data: Dict[str, pd.DataFrame],
        peaks: List[str],
        peak_names: Dict[str, str],
        window_size: int,
        cname_peak_groups: str,
        max_val: float,
        heatmap_height: float,
        figure_height: int,
        panel_width: int,
        title_size: int,
        cmap: str
    ):
        if len(data) == 0:
            return
        if cname_peak_groups != "None" and len(peaks) == 0:
            return

        # Make a vector with the modified names of the selected groups
        if cname_peak_groups != "None":
            window_groups = (
                filtered_windows
                .loc[filtered_windows[cname_peak_groups].isin(peaks)]
                [cname_peak_groups]
                .replace(peak_names)
            )
        else:
            window_groups = None

        half_window = int(window_size / 2.)

        ordered_peaks = []
        for peak in peaks:
            if peak_names[peak] not in ordered_peaks:
                ordered_peaks.append(peak_names[peak])

        fig, axarr = plt.subplots(
            sharex="all",
            sharey="row",
            figsize=(panel_width * len(data), figure_height),
            layout="constrained",
            squeeze=False,
            **format_subplots(data, window_groups, heatmap_height)
        )

        for i, group in enumerate(data.keys()):
            if window_groups is not None:
                df = data[group].loc[window_groups.index]
            else:
                df = data[group]
            df.columns = list(range(
                -half_window,
                half_window,
                int(window_size / df.shape[1])
            ))
            axarr[0, i].set_title(group.replace(" ", "\n"), size=title_size)

            if window_groups is not None:
                for peak_group, peak_group_df in df.groupby(window_groups):
                    plot_density(peak_group_df, axarr[0, i], label=peak_group)
                    row_ix = ordered_peaks.index(peak_group)
                    assert row_ix is not None
                    heatmap = plot_heatmap(peak_group_df, axarr[row_ix + 1, i], max_val, cmap)
                    axarr[row_ix + 1, i].set_ylabel(peak_group, rotation=0, horizontalalignment="right")
            else:
                plot_density(df, axarr[0, i])
                heatmap = plot_heatmap(df, axarr[1, i], max_val, cmap)

            axarr[1, i].set_xticks([0, df.shape[1] / 2., df.shape[1] - 1])
            axarr[1, i].set_xticklabels(
                [
                    "-" + format_bps(half_window),
                    "Center",
                    format_bps(half_window)
                ],
                rotation=90
            )

        fig.colorbar(heatmap, location="bottom", ax=axarr[-1, :], fraction=0.9)
        if cname_peak_groups != "None":
            axarr[0, i].legend(bbox_to_anchor=[1, 1])

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        png_data = buf.read()
        buf.close()

        return fig, png_data


    def format_subplots(
        data: Dict[str, pd.DataFrame],
        window_groups: pd.Series,
        heatmap_height: float,
    ):
        if window_groups is not None:
            # Get the number of peaks in each group
            peak_group_sizes = window_groups.value_counts()

            return dict(
                nrows=1 + window_groups.nunique(),
                ncols=len(data),
                gridspec_kw=dict(
                    height_ratios=[
                        1 - heatmap_height,
                        *[
                            heatmap_height * peak_group_sizes.loc[peak] / peak_group_sizes.sum()
                            for peak in window_groups.unique()
                        ]
                    ],
                    wspace=0.,
                    hspace=0.
                )
            )
        else:
            return dict(
                nrows=2,
                ncols=len(data),
                gridspec_kw=dict(
                    height_ratios=[1 - heatmap_height, heatmap_height],
                    wspace=0.,
                    hspace=0.
                )
            )


    def format_bps(bps: int):
        return f"{bps:,}".replace(",000", "kb")


    def plot_density(
        df: pd.DataFrame,
        ax: Axes,
        label=None
    ):
        density: pd.Series = df.mean().reset_index(drop=True)
        density.plot(ax=ax, label=label)
        axvline(ax, df)


    def plot_heatmap(
        df: pd.DataFrame,
        ax: Axes,
        max_val: float,
        cmap: str
    ):
        plot_df = (
            df
            .loc[df.sum(axis=1).sort_values(ascending=False).index]
            .reset_index(drop=True)
            .clip(upper=max_val)
        )
        heatmap = ax.imshow(
            plot_df,
            aspect="auto",
            cmap=cmap
        )
        ax.set_yticks([])
        axvline(ax, df)
        return heatmap


    def axvline(ax: Axes, df: pd.DataFrame):
        ax.axvline(x=df.shape[1] / 2., linestyle="--", color="black", alpha=0.5)


    _plot_data = plot_data(
        data,
        select_peaks.value.get("groups", []),
        rename_peaks_ui.value,
        window_ui.value['size'],
        **params.value
    )
    if _plot_data is None:
        fig, png_data = None, None
    else:
        fig, png_data = _plot_data
    return fig, png_data


@app.cell
def _(fig):
    fig
    return


@app.cell
def _(mo, png_data):
    mo.download(
        data=png_data,
        filename="tornado_plot.png",
        label="Save as PNG"
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
