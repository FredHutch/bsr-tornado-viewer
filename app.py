import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium", app_title="Marimo Viewer: Cirro")


@app.cell
def _(mo):
    mo.md(r"""# Cut and Run: Tornado Viewer""")
    return


@app.cell
def _():
    # Define the types of datasets which can be read in
    # This is used to filter the dataset selector, below
    cirro_dataset_type_filter = ["custom_dataset"]
    return (cirro_dataset_type_filter,)


@app.cell
def _():
    # Load the marimo library in a dedicated cell for efficiency
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # If the script is running in WASM (instead of local development mode), load micropip
    import sys
    if "pyodide" in sys.modules:
        import micropip
        running_in_wasm = True
    else:
        micropip = None
        running_in_wasm = False
    return micropip, running_in_wasm


@app.cell
async def _(micropip, mo, running_in_wasm):
    with mo.status.spinner("Loading dependencies"):
        # If we are running in WASM, some dependencies need to be set up appropriately.
        # This is really just aligning the needs of the app with the default library versions
        # that come when a marimo app loads in WASM.
        if running_in_wasm:
            print("Installing via micropip")
            # Downgrade plotly to avoid the use of narwhals
            await micropip.install("plotly<6.0.0")
            await micropip.install("ssl")
            micropip.uninstall("urllib3")
            micropip.uninstall("httpx")
            await micropip.install("urllib3==2.3.0")
            micropip.uninstall("requests")
            await micropip.install("requests==2.32.3")
            await micropip.install("httpx==0.26.0")
            await micropip.install("botocore==1.37.3")
            await micropip.install("jmespath==1.0.1")
            await micropip.install("s3transfer==0.11.3")
            await micropip.install("boto3==1.37.3")
            await micropip.install("aiobotocore==2.22.0")
            await micropip.install("cirro[pyodide]==1.5.4")
            await micropip.install("pyBigWig==0.3.24")

        from io import StringIO, BytesIO
        from typing import Dict, List
        from matplotlib import pyplot as plt
        from matplotlib.axes._axes import Axes
        import pandas as pd
        import numpy as np
        from functools import lru_cache
        import pyBigWig
        import tempfile
        from scipy import stats

        from cirro import DataPortalLogin
        from cirro.config import list_tenants

        # A patch to the Cirro client library is applied when running in WASM
        if running_in_wasm:
            from cirro.helpers import pyodide_patch_all
            pyodide_patch_all()

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
def _(list_tenants):
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
def _(domain_to_name, mo, query_params, tenants_by_name):
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
def _(DataPortalLogin, domain_ui, get_client, mo):
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
def _(cirro_login, set_client):
    # Once the user logs in, set the state for the client object
    set_client(cirro_login.await_completion())
    return


@app.cell
def _(get_client, mo):
    # Get the Cirro client object (but only take action if the user selected Cirro as the input)
    client = get_client()
    mo.stop(client is None)
    return (client,)


@app.cell
def _():
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
def _(client):
    # Set the list of projects available to the user
    projects = client.list_projects()
    projects.sort(key=lambda i: i.name)
    return (projects,)


@app.cell
def _(id_to_name, mo, name_to_id, projects, query_params):
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
def _(cirro_dataset_type_filter, client, mo, project_ui):
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
def _(datasets, id_to_name, mo, name_to_id, query_params):
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
def _(client, lru_cache):
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
def _(dataset_ui, list_files, mo, project_ui):
    # Stop if the user has not selected a dataset
    mo.stop(dataset_ui.value is None)

    # Get the list of all files in the selected dataset
    all_files = list_files(project_ui.value, dataset_ui.value)
    return (all_files,)


@app.cell
def _(all_files):
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
def _(filter_files, mo):
    # Ask the user to select a BED file
    select_bed = mo.ui.dropdown(
        label="Select BED file:",
        options=filter_files(suffix=".bed"),
        value=filter_files(suffix=".bed")[-1]
    )
    select_bed
    return (select_bed,)


@app.cell
def _(StringIO, dataset_ui, mo, pd, project_ui, read_file, select_bed):
    # Read in the BED file
    with mo.status.spinner("Reading BED file..."):
        bed = pd.read_csv(
            StringIO(read_file(project_ui.value, dataset_ui.value, select_bed.value)),
            sep="\t",
            header=None
        ).rename(
            columns=dict(zip(range(5), ['chr', 'start', 'end', 'id', 'peak_group']))
        )

    return (bed,)


@app.cell
def _(bed, plt):
    # Show the number of different peak groups
    bed["peak_group"].value_counts().plot(kind="bar")
    plt.ylabel("Number of Peaks")
    plt.xlabel("Peak Group")
    return


@app.cell
def _(filter_files, mo):
    # Ask the user to select one or more bigWig files
    select_bigWigs = mo.ui.multiselect(
        label="Select bigWig file(s):",
        options=filter_files(suffix=".bigWig"),
        value=[]
    )
    select_bigWigs
    return (select_bigWigs,)


@app.cell
def _(download_bigWig, lru_cache, mo):
    # Read in the select_bigWigs files
    @lru_cache
    def read_bigWig(project_id: str, dataset_id: str, fn: str):
        with mo.status.spinner(f"Reading {fn}..."):
            return download_bigWig(project_id, dataset_id, fn)

    return (read_bigWig,)


@app.cell
def _(client, lru_cache, pyBigWig, tempfile):
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
def _(Dict, dataset_ui, project_ui, pyBigWig, read_bigWig, select_bigWigs):
    bigWigs: Dict[str,pyBigWig.pyBigWig] = {
        fn: read_bigWig(project_ui.value, dataset_ui.value, fn)
        for fn in select_bigWigs.value
    }
    return (bigWigs,)


@app.cell
def _(mo, select_bigWigs):
    # Let the user rename and group the wig files
    mo.stop(len(select_bigWigs.value) == 0)
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
        f"name_{sample_ix}": mo.ui.text(label=sample_name, value=sample_name.split("/")[-1][:-len(".bigWig")], full_width=True)
        for sample_ix, sample_name in enumerate(select_bigWigs.value)
    })
    sample_annot_ui
    return (sample_annot_ui,)


@app.cell
def _(client, mo):
    # Get options for how the windows will be set up
    mo.stop(client is None)
    window_ui = mo.md("""
    - {size}
    - {n_bins}
    - {ref}
    - {justification}
    """).batch(
        size=mo.ui.number(label="Window Size:", value=2000),
        n_bins=mo.ui.number(label="Number of Bins:", value=200),
        ref=mo.ui.dropdown(label="Region Reference Point:", options=["Start", "Middle", "End"], value="Middle"),
        justification=mo.ui.dropdown(label="Window Justification:", options=["Left", "Center", "Right"], value="Center")
    )
    window_ui
    return (window_ui,)


@app.cell
def _(bed, np, window_ui):
    # Set up a table with the actual window coordinates
    def _make_windows(size: int, ref: str, justification: str, **kwargs):
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
        )

    windows = _make_windows(**window_ui.value)
    return (windows,)


@app.cell
def _(
    Dict,
    bigWigs: "Dict[str, pyBigWig.pyBigWig]",
    mo,
    np,
    pd,
    stats,
    window_ui,
    windows,
):
    # Compute the windows
    def _get_windows(wig: Dict[str, np.array], r: pd.Series, bar: mo.status.progress_bar, kw: str, n_bins: int, **kwargs):
        bar.update()
        return stats.binned_statistic(
            range(r['window_start'], r['window_end']),
            np.nan_to_num(wig.values(r['chr'], r['window_start'], r['window_end']), nan=0.0),
            'mean',
            bins=n_bins
        ).statistic


    with mo.status.progress_bar(
        title="Calculating window coverage...",
        total=len(bigWigs) * windows.shape[0],
        remove_on_exit=True
    ) as bar:
        window_dfs = {
            kw: pd.DataFrame([
                _get_windows(wig, r, bar, kw, **window_ui.value)
                for _, r in windows.iterrows()
            ], index=windows.index).fillna(0).astype(float)
            for kw, wig in bigWigs.items()
        }
    return (window_dfs,)


@app.cell
def _(mo, sample_annot_ui, window_dfs):
    # Apply labels for each selected wig file and merge replicates
    def _merge_window_data(window_dfs, wig_labels):
        # Make a list of the labels that were applied by the user
        labels = [wig_labels[f"name_{ix}"] for ix in range(len(window_dfs))]

        merged = {}
        for label in labels:
            if label in merged:
                continue
            kws = [kw for kw, _label in zip(window_dfs.keys(), labels) if _label == label]
            if len(kws) == 1:
                merged[label] = window_dfs[kws[0]]
            else:
                merged[label] = sum([window_dfs[kw] for kw in kws]) / len(kws)

        return merged

    with mo.status.spinner("Merging replicates...", remove_on_exit=True):
        data = _merge_window_data(window_dfs, sample_annot_ui.value)

    return (data,)


@app.cell
def _(mo, windows):
    # Select peak groups to display
    _all_peak_groups = windows['peak_group'].drop_duplicates().sort_values().tolist()
    select_peaks = mo.ui.multiselect(
        label="Peak Groups:",
        options=_all_peak_groups,
        value=windows['peak_group'].value_counts().index.values[:1]
    )
    select_peaks
    return (select_peaks,)


@app.cell
def _(mo, select_peaks):
    if len(select_peaks.value) > 0:
        select_peaks_md = "- " + '\n- '.join(list(select_peaks.value))
    else:
        select_peaks_md = "No groups selected"

    mo.md(select_peaks_md)
    return


@app.cell
def _(bigWigs: "Dict[str, pyBigWig.pyBigWig]", data, mo):
    mo.stop(len(bigWigs) == 0)

    params = mo.md("""
    ### Plot Settings

    - {split_peak_groups}
    - {include_samples}
    - {max_val}
    - {heatmap_height}
    - {figure_height}
    - {figure_width},
    - {title_size}
    """).batch(
        include_samples=mo.ui.multiselect(
            label="Include / Reorder Samples:",
            options=sorted(list(data.keys())),
            value=sorted(list(data.keys()))
        ),
        split_peak_groups=mo.ui.checkbox(
            label="Split Peak Groups:",
            value=True
        ),
        max_val=mo.ui.number(
            label="Maximum Value (Heatmap):",
            value=50
        ),
        heatmap_height=mo.ui.number(
            label="Heatmap Height Ratio:",
            start=0.1,
            stop=0.9,
            value=0.75
        ),
        figure_height=mo.ui.number(
            label="Figure Height:",
            start=1,
            stop=100,
            step=1,
            value=6
        ),
        figure_width=mo.ui.number(
            label="Figure Width:",
            start=1,
            stop=100,
            step=1,
            value=6
        ),
        title_size=mo.ui.number(
            label="Panel Font Size:",
            start=1,
            stop=100,
            step=1,
            value=8
        )
    )
    params
    return (params,)


@app.cell
def _(
    Axes,
    BytesIO,
    Dict,
    List,
    data,
    params,
    pd,
    plt,
    select_peaks,
    window_ui,
    windows,
):
    def plot_data(
        data: Dict[str, pd.DataFrame],
        peaks: List[str],
        window_size: int,
        split_peak_groups: bool,
        include_samples: List[str],
        max_val: float,
        heatmap_height: float,
        figure_height: int,
        figure_width: int,
        title_size: int
    ):
        if len(include_samples) == 0:
            return
        if len(data) == 0:
            return
        if len(peaks) == 0:
            return

        half_window = int(window_size / 2.)

        fig, axarr = plt.subplots(
            sharex="all",
            sharey="row",
            figsize=(figure_width, figure_height),
            layout="constrained",
            squeeze=False,
            **format_subplots(data, include_samples, peaks, split_peak_groups, heatmap_height)
        )

        for i, group in enumerate(include_samples):
            df = data[group].loc[
                windows["peak_group"].isin(peaks)
            ]
            df.columns = list(range(
                -half_window,
                half_window,
                int(window_size / df.shape[1])
            ))
            axarr[0, i].set_title(group.replace(" ", "\n"), size=title_size)

            if split_peak_groups:
                for peak_group, peak_group_df in df.groupby(windows["peak_group"]):
                    plot_density(peak_group_df, axarr[0, i], label=peak_group)
                    row_ix = peaks.index(peak_group)
                    assert row_ix is not None
                    heatmap = plot_heatmap(peak_group_df, axarr[row_ix + 1, i], max_val)
                    axarr[row_ix + 1, i].set_ylabel(peak_group, rotation=0, horizontalalignment="right")
            else:
                plot_density(df, axarr[0, i])
                heatmap = plot_heatmap(df, axarr[1, i])

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
        if split_peak_groups:
            axarr[0, i].legend(bbox_to_anchor=[1, 1])

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        png_data = buf.read()
        buf.close()

        return fig, png_data


    def format_subplots(
        data: Dict[str, pd.DataFrame],
        include_samples: List[str],
        peaks: List[str],
        split_peak_groups: bool,
        heatmap_height: float,
    ):
        if split_peak_groups:
            # Get the number of peaks in each group
            peak_group_sizes = windows.groupby("peak_group").apply(len).loc[peaks]

            return dict(
                nrows=1 + len(peaks),
                ncols=len(include_samples),
                gridspec_kw=dict(
                    height_ratios=[
                        1 - heatmap_height,
                        *[
                            heatmap_height * peak_group_sizes.loc[peak] / peak_group_sizes.sum()
                            for peak in peaks
                        ]
                    ],
                    wspace=0.,
                    hspace=0.
                )
            )
        else:
            return dict(
                nrows=2,
                ncols=len(include_samples),
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
        max_val: float
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
            cmap="Blues"
        )
        ax.set_yticks([])
        axvline(ax, df)
        return heatmap


    def axvline(ax: Axes, df: pd.DataFrame):
        ax.axvline(x=df.shape[1] / 2., linestyle="--", color="black", alpha=0.5)


    fig, png_data = plot_data(
        data,
        select_peaks.value,
        window_ui.value['size'],
        **params.value
    )
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
