args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
if (length(file_arg) > 0) {
  script_path <- normalizePath(
    sub("^--file=", "", file_arg[[1]]),
    winslash = "/",
    mustWork = TRUE
  )
  script_dir <- dirname(script_path)
} else {
  script_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

figure_root <- if (basename(script_dir) == "code") {
  dirname(script_dir)
} else {
  script_dir
}
data_dir <- file.path(figure_root, "data")
figures_dir <- file.path(figure_root, "figures")

local_r_lib <- file.path(figure_root, "library")
if (dir.exists(local_r_lib)) {
  .libPaths(c(local_r_lib, .libPaths()))
}

required_packages <- c("ggplot2", "dplyr", "tikzDevice", "scales", "patchwork")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(
    "Missing required R packages: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tikzDevice)
  library(patchwork)
})

require_columns <- function(df, required_cols, context) {
  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) {
    stop(
      context,
      ": missing columns ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
}

compile_tikz_pdf <- function(tex_path) {
  old_wd <- getwd()
  tex_dir <- dirname(tex_path)
  tex_file <- basename(tex_path)
  pdf_path <- sub("\\.tex$", ".pdf", tex_path)
  aux_path <- sub("\\.tex$", ".aux", tex_path)
  log_path <- sub("\\.tex$", ".log", tex_path)

  on.exit(setwd(old_wd), add = TRUE)
  setwd(tex_dir)

  compile_output <- system2(
    "pdflatex",
    c("-interaction=nonstopmode", "-halt-on-error", tex_file),
    stdout = TRUE,
    stderr = TRUE
  )
  status <- attr(compile_output, "status")
  if (!is.null(status) && status != 0) {
    stop(
      "pdflatex failed for ",
      tex_path,
      ":\n",
      paste(compile_output, collapse = "\n"),
      call. = FALSE
    )
  }
  if (!file.exists(pdf_path)) {
    stop("Expected compiled PDF not found: ", pdf_path, call. = FALSE)
  }

  cropped_pdf_file <- sub("\\.pdf$", "_cropped.pdf", basename(pdf_path))
  crop_output <- system2(
    "pdfcrop",
    c("--margins", "0", basename(pdf_path), cropped_pdf_file),
    stdout = TRUE,
    stderr = TRUE
  )
  crop_status <- attr(crop_output, "status")
  if (!is.null(crop_status) && crop_status != 0) {
    stop(
      "pdfcrop failed for ",
      pdf_path,
      ":\n",
      paste(crop_output, collapse = "\n"),
      call. = FALSE
    )
  }
  cropped_pdf_path <- file.path(tex_dir, cropped_pdf_file)
  if (!file.exists(cropped_pdf_path)) {
    stop("Expected cropped PDF not found: ", cropped_pdf_path, call. = FALSE)
  }
  if (!file.rename(cropped_pdf_path, pdf_path)) {
    stop("Failed to replace PDF with cropped output: ", pdf_path, call. = FALSE)
  }

  unlink(c(aux_path, log_path), force = TRUE)
  pdf_path
}

save_plot <- function(plot, pdf_path, width, height) {
  if (!grepl("\\.pdf$", pdf_path, ignore.case = TRUE)) {
    stop("Output path must end in .pdf: ", pdf_path, call. = FALSE)
  }

  tex_path <- sub("\\.pdf$", ".tex", pdf_path, ignore.case = TRUE)
  dir.create(dirname(pdf_path), recursive = TRUE, showWarnings = FALSE)

  tikzDevice::tikz(
    file = tex_path,
    width = width,
    height = height,
    standAlone = TRUE,
    sanitize = FALSE,
    engine = "pdftex",
    verbose = FALSE
  )
  print(plot)
  dev.off()

  compile_tikz_pdf(tex_path)
  message("Wrote ", pdf_path)
}

paper_theme <- function() {
  theme_bw(base_size = 10, base_family = "serif") +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text.x = element_text(color = "black", size = 9),
      axis.text.y = element_text(color = "black", size = 9),
      axis.title = element_text(size = 11),
      axis.title.y = element_text(margin = margin(r = 2)),
      strip.text = element_text(
        size = 10,
        face = "bold",
        vjust = 0.5,
        margin = margin(t = 0, b = 2)
      ),
      strip.clip = "off",
      strip.background = element_rect(fill = "white", color = NA),
      panel.spacing.x = grid::unit(0.15, "lines"),
      legend.position = "none",
      plot.margin = margin(1, -3, -4, -3)
    )
}

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1) {
  normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
} else {
  file.path(data_dir, "backbone_scenario_heatmap_data.csv")
}
output_path <- if (length(args) >= 2) {
  normalizePath(args[[2]], winslash = "/", mustWork = FALSE)
} else {
  file.path(figures_dir, "Figure_2_backbone_scenario_heatmap.pdf")
}
if (length(args) > 2) {
  stop("Expected at most two arguments: [input_csv] [output_pdf].", call. = FALSE)
}

heatmap_raw <- read.csv(
  input_path,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  na.strings = c("", "NA")
)
require_columns(
  heatmap_raw,
  c(
    "dataset",
    "backbone",
    "scenario",
    "D",
    "dataset_label",
    "scenario_label"
  ),
  basename(input_path)
)

heatmap_raw <- heatmap_raw %>%
  rename(
    model_architecture = backbone,
    dataset_display = dataset_label,
    scenario_display = scenario_label
  )

dataset_order <- c(
  "Beijing Air Tiantan",
  "Penmanshiel WT08",
  "ETTh1",
  "Traffic"
)
dataset_display_map <- c(
  "Beijing Air Tiantan" = "Beijing Air Tiantan",
  "Penmanshiel WT08" = "Penmanshiel WT08",
  "ETTh1" = "ETTh1",
  "traffic" = "Traffic",
  "Traffic" = "Traffic"
)
architecture_order <- c(
  "SeasonalNaive",
  "DLinear",
  "GRU",
  "ModernTCN",
  "TSMixer",
  "PatchTST",
  "Chronos2"
)
architecture_labels <- c(
  "SeasonalNaive" = "SNaive",
  "DLinear" = "DLinear",
  "GRU" = "GRU",
  "ModernTCN" = "ModTCN",
  "TSMixer" = "TSMixer",
  "PatchTST" = "PatchTST",
  "Chronos2" = "Chronos2"
)
scenario_order <- c(
  "drift",
  "attenuation",
  "noise",
  "spike",
  "time_stretch",
  "time_compress",
  "stuck_sensor",
  "missing_data"
)
scenario_labels <- c(
  "drift" = "Drift",
  "attenuation" = "Attenuation",
  "noise" = "Noise",
  "spike" = "Spike",
  "time_stretch" = "TimeStretch",
  "time_compress" = "TimeCompress",
  "stuck_sensor" = "StuckSensor",
  "missing_data" = "MissingData"
)
architecture_positions <- stats::setNames(
  rev(seq_along(architecture_order)),
  architecture_order
)

heatmap_df <- heatmap_raw %>%
  mutate(
    dataset_display = unname(dataset_display_map[dataset_display]),
    scenario_display = unname(scenario_labels[scenario]),
    scenario_idx = match(scenario, .env$scenario_order),
    architecture_y = unname(architecture_positions[model_architecture])
  )

if (any(is.na(heatmap_df$dataset_display))) {
  unknown_datasets <- unique(heatmap_raw$dataset_display[is.na(heatmap_df$dataset_display)])
  stop(
    "Unsupported dataset_display values: ",
    paste(unknown_datasets, collapse = ", "),
    call. = FALSE
  )
}
if (any(is.na(heatmap_df$scenario_display))) {
  unknown_scenarios <- unique(heatmap_raw$scenario[is.na(heatmap_df$scenario_display)])
  stop(
    "Unsupported scenario values: ",
    paste(unknown_scenarios, collapse = ", "),
    call. = FALSE
  )
}
if (any(is.na(heatmap_df$architecture_y))) {
  unknown_architectures <- unique(
    heatmap_raw$model_architecture[is.na(heatmap_df$architecture_y)]
  )
  stop(
    "Unsupported model_architecture values: ",
    paste(unknown_architectures, collapse = ", "),
    call. = FALSE
  )
}

expected_grid <- expand.grid(
  dataset_display = dataset_order,
  model_architecture = architecture_order,
  scenario = scenario_order,
  stringsAsFactors = FALSE
)
coverage_df <- expected_grid %>%
  left_join(
    heatmap_df %>%
      select(dataset_display, model_architecture, scenario, D),
    by = c("dataset_display", "model_architecture", "scenario")
  )
if (any(is.na(coverage_df$D))) {
  missing_cells <- coverage_df %>%
    filter(is.na(D)) %>%
    transmute(
      label = paste(dataset_display, model_architecture, scenario, sep = " / ")
    )
  stop(
    "Missing heatmap values for: ",
    paste(missing_cells$label, collapse = ", "),
    call. = FALSE
  )
}

if (anyDuplicated(heatmap_df[c("dataset_display", "model_architecture", "scenario")]) > 0) {
  stop(
    "Duplicate dataset/model/scenario rows are not allowed in ",
    basename(input_path),
    ".",
    call. = FALSE
  )
}

heatmap_df <- heatmap_df %>%
  mutate(
    dataset_display = factor(dataset_display, levels = .env$dataset_order),
    D_label = formatC(D, format = "f", digits = 2),
    D_label_color = if_else(D >= 1.45, "white", "black")
  ) %>%
  arrange(dataset_display, model_architecture, scenario_idx)

fill_lower <- min(
  1,
  floor(min(heatmap_df$D, na.rm = TRUE) * 20) / 20
)
fill_limits <- c(fill_lower, 2)
legend_breaks <- c(1, 1.5, 2)
legend_breaks <- legend_breaks[
  legend_breaks >= fill_limits[[1]] & legend_breaks <= fill_limits[[2]]
]

main_plot <- ggplot(
  heatmap_df,
  aes(x = scenario_idx, y = architecture_y, fill = D)
) +
  geom_tile(
    width = 0.98,
    height = 0.98,
    color = "#F3F3F3",
    linewidth = 0.3
  ) +
  geom_text(
    aes(label = D_label, color = D_label_color),
    family = "serif",
    size = 1.7
  ) +
  facet_grid(. ~ dataset_display) +
  scale_x_continuous(
    breaks = seq_along(scenario_order),
    labels = unname(scenario_labels[scenario_order]),
    expand = expansion(mult = c(0, 0), add = c(0.02, 0.02))
  ) +
  scale_y_continuous(
    breaks = unname(architecture_positions[architecture_order]),
    labels = unname(architecture_labels[architecture_order]),
    limits = c(0.5, 7.5),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_fill_gradient2(
    low = "#FFFFFF",
    mid = "#F3D5C7",
    high = "#D92523",
    midpoint = 1,
    limits = fill_limits,
    oob = scales::squish,
    guide = "none"
  ) +
  scale_color_identity(guide = "none") +
  coord_fixed(ratio = 1, clip = "off") +
  labs(x = NULL, y = "Architecture") +
  paper_theme() +
  theme(
    axis.text.x = element_text(
      angle = 30,
      hjust = 1,
      vjust = 1,
      color = "black",
      size = 7.1,
      lineheight = 0.82
    ),
    axis.text.y = element_text(color = "black", size = 9.2),
    panel.border = element_blank(),
    strip.text.x = element_text(vjust = 0.5, margin = margin(t = 0, b = 2))
  )

if (fill_limits[[1]] < 1) {
  legend_pal <- scales::gradient_n_pal(
    colours = c("#FFFFFF", "#F3D5C7", "#D92523"),
    values = scales::rescale(c(fill_limits[[1]], 1, fill_limits[[2]]), from = fill_limits)
  )
} else {
  legend_pal <- scales::gradient_n_pal(
    colours = c("#F3D5C7", "#D92523"),
    values = c(0, 1)
  )
}
legend_cols <- legend_pal(seq(0, 1, length.out = 1024))
legend_raster_vertical <- as.raster(matrix(rev(legend_cols), ncol = 1))

legend_plot_right <- ggplot() +
  annotation_raster(
    legend_raster_vertical,
    xmin = 0.665,
    xmax = 0.720,
    ymin = fill_limits[[1]],
    ymax = fill_limits[[2]]
  ) +
  annotate(
    "text",
    x = 0.69,
    y = fill_limits[[2]] + 0.10,
    label = "$\\mathcal{D}_p$",
    family = "serif",
    hjust = 0.5,
    vjust = 0,
    size = 4.6
  ) +
  annotate(
    "text",
    x = 0.730,
    y = legend_breaks,
    label = formatC(legend_breaks, format = "f", digits = 1),
    family = "serif",
    hjust = 0,
    size = 3.4
  ) +
  coord_cartesian(
    xlim = c(0.65, 0.765),
    ylim = c(fill_limits[[1]] - 0.02, fill_limits[[2]] + 0.11),
    expand = FALSE,
    clip = "off"
  ) +
  theme_void(base_size = 10, base_family = "serif") +
  theme(
    aspect.ratio = 6,
    plot.margin = margin(0, 0, 0, -28),
    plot.background = element_rect(fill = "white", color = NA)
  )

main_plot_right_legend <- main_plot +
  labs(x = "Scenario") +
  theme(
    axis.title.x = element_text(size = 11, margin = margin(t = 2)),
    plot.margin = margin(1, -16, -2, -3)
  )

legend_column_right <- (
  patchwork::plot_spacer() /
    legend_plot_right /
    patchwork::plot_spacer()
) +
  patchwork::plot_layout(heights = c(0.14, 0.66, 0.20))

final_plot_right_legend <- (
  main_plot_right_legend |
    legend_column_right
) +
  patchwork::plot_layout(widths = c(1, 0.045))

save_plot(final_plot_right_legend, output_path, width = 7.45, height = 3.0)
