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

        from io import StringIO, BytesIO
        from queue import Queue
        from time import sleep
        from typing import Dict, Optional, List
        import plotly.express as px
        from matplotlib import pyplot as plt
        from matplotlib.axes._axes import Axes
        import pandas as pd
        import numpy as np
        from functools import lru_cache
        import base64
        from urllib.parse import quote_plus

        from cirro import DataPortalLogin, DataPortalDataset
        from cirro.config import list_tenants

        # A patch to the Cirro client library is applied when running in WASM
        if running_in_wasm:
            from cirro.helpers import pyodide_patch_all
            pyodide_patch_all()

    return (
        Axes,
        BytesIO,
        DataPortalDataset,
        DataPortalLogin,
        Dict,
        List,
        list_tenants,
        lru_cache,
        pd,
        plt,
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
def _(DataPortalDataset, Dict, client, dataset_ui, lru_cache, mo, pd):
    # Stop if the user has not selected a dataset
    mo.stop(dataset_ui.value is None)

    # Parse the dataset
    class PeakData:

        msg: str
        valid_dataset: bool
        metadata: pd.DataFrame
        peak_dfs: Dict[str, pd.DataFrame]

        def __init__(
            self,
            ds: DataPortalDataset
        ):
            # Read all of the CSV files
            self.dfs = {}
            for file in ds.list_files():
                if file.name.endswith((".csv", ".csv.gz")):
                    print(f"Reading {file.name}")
                    try:
                        self.dfs[file.name.replace("data/", "")] = file.read_csv()
                    except Exception as e:
                        self.msg = f"Reading {file.name}\n\n" + str(e)
                        self.valid_dataset = False
                        break

            # Find the metadata table
            if self.has_metadata():
                if self.has_peak_data():
                    self.valid_dataset = True
                    self.msg = f"Read in {len(self.peak_dfs):,} datasets for {self.metadata.shape[0]:,} peaks"
                else:
                    self.valid_dataset = False
                    self.msg = "No peak data found"
            else:
                self.valid_dataset = False
                self.msg = "No metadata file found."

        def has_peak_data(self):
            self.peak_dfs = {}

            for file_name, df in self.dfs.items():
                if df.shape[0] == self.metadata.shape[0]:
                    if all([cname.startswith(("u", "d")) for cname in df.columns.values]):
                        self.peak_dfs[file_name] = df

            # Do some fancy parsing of the filenames to figure out good labels
            self.rename_peak_dfs()

            return len(self.peak_dfs) > 0

        def rename_peak_dfs(self):
            if len(self.peak_dfs) <= 1:
                return

            # Break up each file into fields
            fields = {}
            for file_name in self.peak_dfs:
                for field in file_name.split("."):
                    fields[field] = fields.get(field, 0) + 1

            name_map = {}
            for file_name in self.peak_dfs:
                unique_fields = " ".join([
                    field.replace("_", " ")
                    for field in file_name.split(".")
                    if fields[field] < len(self.peak_dfs)
                ])
                if len(unique_fields) > 0:
                    name_map[file_name] = unique_fields

            self.peak_dfs = {
                name_map.get(file_name, file_name): df
                for file_name, df in self.peak_dfs.items()
            }

        def has_metadata(self):
            for df in self.dfs.values():
                if all([
                    cname in df.columns.values
                    for cname in [
                        "chrom",
                        "start",
                        "end",
                        "peak_id",
                        "sample_groups",
                        "peak_no",
                        "peak_group"
                    ]
                ]):
                    self.metadata = df
                    return True
            return False


    @lru_cache
    def read_peak_data(project_id: str, dataset_id: str):
        return PeakData(
            (
                client
                .get_project_by_id(project_id)
                .get_dataset_by_id(dataset_id)
            )
        )

    return PeakData, read_peak_data


@app.cell
def _(dataset_ui, mo, project_ui, read_peak_data):
    data = read_peak_data(project_ui.value, dataset_ui.value)
    mo.md(data.msg)
    return (data,)


@app.cell
def _(data, mo):
    # Select sample groups to display
    select_groups = mo.ui.multiselect(
        label="Sample Groups:",
        options=list(data.peak_dfs.keys()),
        value=list(data.peak_dfs.keys())
    )
    select_groups
    return (select_groups,)


@app.cell
def _(mo, select_groups):
    if len(select_groups.value) > 0:
        select_groups_md = "- " + '\n- '.join(list(select_groups.value))
    else:
        select_groups_md = "No groups selected"

    mo.md(select_groups_md)
    return


@app.cell
def _(data, mo):
    # Select peak groups to display
    _all_peak_groups = data.metadata['peak_group'].drop_duplicates().sort_values().tolist()
    select_peaks = mo.ui.multiselect(
        label="Peak Groups:",
        options=_all_peak_groups,
        value=_all_peak_groups
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
def _(data, mo):
    mo.stop(data.valid_dataset is False)

    params = mo.md("""
    ### Plot Settings

    - {split_peak_groups}
    - {window_size}
    - {clip_quantile}
    - {heatmap_height}
    - {figure_height}
    - {figure_width},
    - {title_size}
    """).batch(
        split_peak_groups=mo.ui.checkbox(
            label="Split Peak Groups:",
            value=True
        ),
        window_size=mo.ui.number(
            label="Window Size (bp):",
            start=100,
            step=100,
            value=2000
        ),
        heatmap_height=mo.ui.number(
            label="Heatmap Height Ratio:",
            start=0.1,
            stop=0.9,
            value=0.75
        ),
        clip_quantile=mo.ui.number(
            label="Clip Quantile:",
            start=0.1,
            stop=1.0,
            step=0.01,
            value=0.8
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
    List,
    PeakData,
    data,
    params,
    pd,
    plt,
    select_groups,
    select_peaks,
):
    def plot_data(
        data: PeakData,
        groups: List[str],
        peaks: List[str],
        split_peak_groups: bool,
        window_size: int,
        clip_quantile: float,
        heatmap_height: float,
        figure_height: int,
        figure_width: int,
        title_size: int
    ):
        if len(groups) == 0:
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
            **format_subplots(data, groups, peaks, split_peak_groups, heatmap_height)
        )

        for i, group in enumerate(groups):
            df = data.peak_dfs[group].loc[
                data.metadata["peak_group"].isin(peaks)
            ]
            df.columns = list(range(
                -half_window,
                half_window,
                int(window_size / df.shape[1])
            ))
            axarr[0, i].set_title(group.replace(" ", "\n"), size=title_size)

            if split_peak_groups:
                for peak_group, peak_group_df in df.groupby(data.metadata["peak_group"]):
                    plot_density(peak_group_df, axarr[0, i], label=peak_group)
                    row_ix = peaks.index(peak_group)
                    assert row_ix is not None
                    heatmap = plot_heatmap(peak_group_df, axarr[row_ix + 1, i], clip_quantile)
                    axarr[row_ix + 1, i].set_ylabel(peak_group, rotation=0, horizontalalignment="right")
            else:
                plot_density(df, axarr[0, i])
                heatmap = plot_heatmap(df, axarr[1, i], clip_quantile)

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
        data: PeakData,
        groups: List[str],
        peaks: List[str],
        split_peak_groups: bool,
        heatmap_height: float,
    ):
        if split_peak_groups:
            # Get the number of peaks in each group
            peak_group_sizes = data.metadata.groupby("peak_group").apply(len).loc[peaks]

            return dict(
                nrows=1 + len(peaks),
                ncols=len(groups),
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
                ncols=len(groups),
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
        clip_quantile: float
    ):
        heatmap = ax.imshow(
            (
                df
                .loc[df.sum(axis=1).sort_values(ascending=False).index]
                .reset_index(drop=True)
                .clip(
                    upper=df.quantile(q=clip_quantile).max()
                )
            ),
            aspect="auto",
            cmap="RdYlBu"
        )
        ax.set_yticks([])
        axvline(ax, df)
        return heatmap


    def axvline(ax: Axes, df: pd.DataFrame):
        ax.axvline(x=df.shape[1] / 2., linestyle="--", color="black", alpha=0.5)


    fig, png_data = plot_data(
        data,
        select_groups.value,
        select_peaks.value,
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
