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

required_packages <- c("ggplot2", "dplyr", "ggrepel", "ggh4x", "tikzDevice")
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
  library(ggrepel)
  library(ggh4x)
  library(tikzDevice)
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

  unlink(c(aux_path, log_path), force = TRUE)
  pdf_path
}

save_plot <- function(plot, pdf_path, width, height) {
  if (!grepl("\\.pdf$", pdf_path, ignore.case = TRUE)) {
    stop("Output path must end in .pdf: ", pdf_path, call. = FALSE)
  }

  tex_path <- sub("\\.pdf$", ".tex", pdf_path, ignore.case = TRUE)
  out_dir <- dirname(pdf_path)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

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

pareto_frontier <- function(df, x_col, y_col) {
  if (nrow(df) == 0) {
    return(df[0, , drop = FALSE])
  }

  keep <- rep(TRUE, nrow(df))
  x_values <- df[[x_col]]
  y_values <- df[[y_col]]

  for (idx in seq_len(nrow(df))) {
    dominated <- (x_values <= x_values[[idx]]) &
      (y_values <= y_values[[idx]]) &
      ((x_values < x_values[[idx]]) | (y_values < y_values[[idx]]))
    dominated[[idx]] <- FALSE
    if (any(dominated)) {
      keep[[idx]] <- FALSE
    }
  }

  frontier <- df[keep, , drop = FALSE]
  frontier[order(frontier[[x_col]], frontier[[y_col]]), , drop = FALSE]
}

frontier_by_dataset <- function(df, x_col, y_col) {
  split_df <- split(df, df$panel_title, drop = TRUE)
  dplyr::bind_rows(lapply(split_df, pareto_frontier, x_col = x_col, y_col = y_col))
}

paper_theme <- function() {
  theme_bw(base_size = 10, base_family = "serif") +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      strip.background = element_blank(),
      axis.text.x = element_text(color = "black", size = 9),
      axis.text.y = element_text(color = "black", size = 9),
      strip.text = element_text(size = 10, face = "bold"),
      legend.position = "none",
      axis.title = element_text(size = 11),
      plot.margin = margin(0, 0, 0, 0)
    )
}

method_labels <- c(
  "adversarial_training" = "+ PGD Adv.\nTraining"
)

backbone_family_map <- c(
  "PatchTST" = "Attention",
  "ModernTCN" = "Convolution",
  "Chronos2" = "Foundation",
  "DLinear" = "Fully Connected",
  "TSMixer" = "Fully Connected",
  "GRU" = "Recurrent",
  "SeasonalNaive" = "Statistical"
)

backbone_family_palette <- c(
  "Attention" = "#0072B2",
  "Convolution" = "#D55E00",
  "Foundation" = "#009E73",
  "Fully Connected" = "#E69F00",
  "Recurrent" = "#56B4E9",
  "Statistical" = "#CC79A7"
)

panel_title_labels <- c(
  "BeijingAir" = "Beijing Air Tiantan",
  "Penmanshiel" = "Penmanshiel WT08",
  "ETTh1" = "ETTh1",
  "Traffic" = "Traffic"
)

args <- commandArgs(trailingOnly = TRUE)
output_path <- if (length(args) >= 1) {
  normalizePath(args[[1]], winslash = "/", mustWork = FALSE)
} else {
  file.path(figures_dir, "Figure_6_trajectories.pdf")
}

trajectory_path <- file.path(data_dir, "pgd_trajectory_data.csv")

if (!file.exists(trajectory_path)) {
  stop("Missing local input file: ", trajectory_path, call. = FALSE)
}

trajectory_raw <- read.csv(
  trajectory_path,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  na.strings = c("", "NA")
)

require_columns(
  trajectory_raw,
  c(
    "dataset_order",
    "dataset",
    "dataset_label",
    "backbone",
    "architecture_family",
    "improvement_method",
    "baseline_MSE_c",
    "baseline_D_w",
    "improvement_MSE_c",
    "improvement_D_w"
  ),
  basename(trajectory_path)
)

trajectory_raw <- trajectory_raw %>%
  transmute(
    panel_order = dataset_order,
    panel_title = ifelse(dataset_label == "traffic", "Traffic", dataset_label),
    dataset = dataset,
    backbone_architecture = backbone,
    architecture_family = architecture_family,
    robustness_method = improvement_method,
    baseline_MSE_test = baseline_MSE_c,
    baseline_D_w = baseline_D_w,
    improved_MSE_test = improvement_MSE_c,
    improved_D_w = improvement_D_w
  )

unknown_methods <- setdiff(unique(trajectory_raw$robustness_method), names(method_labels))
if (length(unknown_methods) > 0) {
  stop(
    basename(trajectory_path),
    ": unsupported robustness_method values ",
    paste(unknown_methods, collapse = ", "),
    call. = FALSE
  )
}

architectures_in_plot <- sort(unique(trajectory_raw$backbone_architecture))
unknown_architectures <- setdiff(architectures_in_plot, names(backbone_family_map))
if (length(unknown_architectures) > 0) {
  stop(
    basename(trajectory_path),
    ": unsupported backbone_architecture values ",
    paste(unknown_architectures, collapse = ", "),
    call. = FALSE
  )
}

panel_levels <- trajectory_raw %>%
  distinct(panel_order, panel_title) %>%
  arrange(panel_order) %>%
  pull(panel_title)

baseline_df <- trajectory_raw %>%
  distinct(
    dataset_id = dataset,
    panel_order,
    panel_title,
    architecture = backbone_architecture,
    baseline_MSE_test,
    baseline_D_w
  ) %>%
  mutate(
    panel_title = factor(panel_title, levels = panel_levels),
    architecture = factor(architecture, levels = names(backbone_family_map)),
    architecture_family = factor(
      unname(backbone_family_map[as.character(architecture)]),
      levels = names(backbone_family_palette)
    )
  )

expected_pairs <- nrow(
  trajectory_raw %>%
    distinct(dataset, backbone_architecture)
)
if (nrow(baseline_df) != expected_pairs) {
  stop(
    "Failed to resolve all baseline points needed for the trajectory plot.",
    call. = FALSE
  )
}

trajectory_df <- trajectory_raw %>%
  transmute(
    panel_order = panel_order,
    panel_title = factor(panel_title, levels = panel_levels),
    dataset_id = dataset,
    architecture = factor(backbone_architecture, levels = names(backbone_family_map)),
    architecture_family = factor(
      unname(backbone_family_map[backbone_architecture]),
      levels = names(backbone_family_palette)
    ),
    improvement_label = unname(method_labels[robustness_method]),
    baseline_MSE_test = baseline_MSE_test,
    baseline_D_w = baseline_D_w,
    improved_MSE_test = improved_MSE_test,
    improved_D_w = improved_D_w
  ) %>%
  arrange(panel_order, architecture)

improvement_labels <- unique(as.character(trajectory_df$improvement_label))
if (length(improvement_labels) != 1) {
  stop(
    "trajectory.csv must contain exactly one robustness method for this figure.",
    call. = FALSE
  )
}
improvement_label_value <- improvement_labels[[1]]
baseline_legend_label <- "Baseline model"
shape_values <- c("Baseline model" = 16)
shape_values[[improvement_label_value]] <- 21

baseline_points <- baseline_df %>%
  transmute(
    panel_title = panel_title,
    dataset_id = dataset_id,
    architecture = architecture,
    architecture_family = architecture_family,
    MSE_test = baseline_MSE_test,
    D_w = baseline_D_w
  )

panel_x_specs <- data.frame(
  panel_title = c("Beijing Air Tiantan", "Penmanshiel WT08", "ETTh1", "Traffic"),
  x_min = c(0.276, 0.3335, 0.41, 0.39),
  x_max = c(0.364, 0.3565, 0.69, 0.77),
  stringsAsFactors = FALSE
)

y_points <- c(baseline_points$D_w, trajectory_df$improved_D_w)
y_min <- min(y_points, na.rm = TRUE)
y_max <- max(y_points, na.rm = TRUE)
y_span <- y_max - y_min
if (!is.finite(y_span) || y_span <= 0) {
  stop("Failed to resolve shared y-axis span for arrow geometry.", call. = FALSE)
}

arrow_df <- trajectory_df %>%
  inner_join(
    baseline_points %>%
      transmute(
        panel_title = panel_title,
        dataset_id = dataset_id,
        architecture = architecture,
        arrow_xstart = MSE_test,
        arrow_ystart = D_w
      ),
    by = c("panel_title", "dataset_id", "architecture")
  ) %>%
  inner_join(panel_x_specs, by = "panel_title") %>%
  mutate(
    panel_title = factor(panel_title, levels = panel_levels),
    x_span = x_max - x_min,
    dx_norm = (improved_MSE_test - arrow_xstart) / x_span,
    dy_norm = (improved_D_w - arrow_ystart) / y_span,
    segment_norm = sqrt(dx_norm^2 + dy_norm^2),
    safe_segment_norm = ifelse(segment_norm > 0, segment_norm, 1),
    ring_gap_norm = 0.032,
    min_arrow_segment_norm = 0.050,
    gap_norm = pmin(ring_gap_norm, pmax(segment_norm - 1e-6, 0)),
    arrow_xend = improved_MSE_test - gap_norm * (dx_norm / safe_segment_norm) * x_span,
    arrow_yend = improved_D_w - gap_norm * (dy_norm / safe_segment_norm) * y_span,
    line_only = (
      (panel_title == "Traffic" & architecture == "GRU") |
      (panel_title == "Penmanshiel WT08" & architecture == "DLinear")
    )
  ) %>%
  filter(segment_norm > min_arrow_segment_norm)

improved_points <- trajectory_df %>%
  transmute(
    panel_title = panel_title,
    dataset_id = dataset_id,
    architecture = architecture,
    architecture_family = architecture_family,
    MSE_test = improved_MSE_test,
    D_w = improved_D_w
  )

label_points <- baseline_points %>%
  mutate(
    label_nudge_x = 0,
    label_nudge_y = 0,
    label_nudge_x = ifelse(panel_title == "Beijing Air Tiantan" & architecture == "GRU", 0.006, label_nudge_x),
    label_nudge_y = ifelse(panel_title == "Beijing Air Tiantan" & architecture == "GRU", 0.035, label_nudge_y),
    label_nudge_y = ifelse(panel_title == "Beijing Air Tiantan" & architecture == "ModernTCN", 0.04, label_nudge_y),
    label_nudge_x = ifelse(panel_title == "Beijing Air Tiantan" & architecture == "PatchTST", 0.005, label_nudge_x),
    label_nudge_y = ifelse(panel_title == "Beijing Air Tiantan" & architecture == "PatchTST", 0.04, label_nudge_y),
    label_nudge_y = ifelse(panel_title == "Beijing Air Tiantan" & architecture == "TSMixer", +0.04, label_nudge_y),
    label_nudge_y = ifelse(panel_title == "Beijing Air Tiantan" & architecture == "DLinear", 0.04, label_nudge_y),
    label_nudge_x = ifelse(panel_title == "Penmanshiel WT08" & architecture == "TSMixer", 0.006, label_nudge_x),
    label_nudge_y = ifelse(panel_title == "Penmanshiel WT08" & architecture == "TSMixer", 0.035, label_nudge_y),
    label_nudge_x = ifelse(panel_title == "Penmanshiel WT08" & architecture == "ModernTCN", 0.0043, label_nudge_x),
    label_nudge_y = ifelse(panel_title == "Penmanshiel WT08" & architecture == "ModernTCN", 0.008, label_nudge_y),
    label_nudge_y = ifelse(panel_title == "Penmanshiel WT08" & architecture == "DLinear", 0.04, label_nudge_y),
    label_nudge_x = ifelse(panel_title == "Penmanshiel WT08" & architecture == "PatchTST", 0.013, label_nudge_x),
    label_nudge_y = ifelse(panel_title == "Penmanshiel WT08" & architecture == "PatchTST", -0.038, label_nudge_y),
    label_nudge_x = ifelse(panel_title == "Penmanshiel WT08" & architecture == "GRU", 0.001, label_nudge_x),
    label_nudge_y = ifelse(panel_title == "Penmanshiel WT08" & architecture == "GRU", 0.035, label_nudge_y),
    label_nudge_x = ifelse(panel_title == "ETTh1" & architecture == "ModernTCN", 0.06, label_nudge_x),
    label_nudge_y = ifelse(panel_title == "ETTh1" & architecture == "ModernTCN", 0.02, label_nudge_y),
    label_nudge_y = ifelse(panel_title == "ETTh1" & architecture == "GRU", 0.04, label_nudge_y),
    label_nudge_x = ifelse(panel_title == "ETTh1" & architecture == "PatchTST", 0.031, label_nudge_x),
    label_nudge_y = ifelse(panel_title == "ETTh1" & architecture == "PatchTST", 0.04, label_nudge_y),
    label_nudge_x = ifelse(panel_title == "ETTh1" & architecture == "DLinear", 0.051, label_nudge_x),
    label_nudge_y = ifelse(panel_title == "ETTh1" & architecture == "DLinear", 0.01, label_nudge_y),
    label_nudge_x = ifelse(panel_title == "ETTh1" & architecture == "TSMixer", 0.035, label_nudge_x),
    label_nudge_y = ifelse(panel_title == "ETTh1" & architecture == "TSMixer", 0.02, label_nudge_y),
    label_nudge_x = ifelse(panel_title == "Traffic" & architecture == "PatchTST", 0.06, label_nudge_x),
    label_nudge_y = ifelse(panel_title == "Traffic" & architecture == "PatchTST", 0.03, label_nudge_y),
    label_nudge_x = ifelse(panel_title == "Traffic" & architecture == "ModernTCN", 0.1, label_nudge_x),
    label_nudge_y = ifelse(panel_title == "Traffic" & architecture == "ModernTCN", -0.04, label_nudge_y),
    label_nudge_x = ifelse(panel_title == "Traffic" & architecture == "DLinear", 0.12, label_nudge_x),
    label_nudge_x = ifelse(panel_title == "Traffic" & architecture == "TSMixer", 0.015, label_nudge_x),
    label_nudge_y = ifelse(panel_title == "Traffic" & architecture == "TSMixer", 0.05, label_nudge_y),
    label_nudge_x = ifelse(panel_title == "Traffic" & architecture == "GRU", 0.01, label_nudge_x),
    label_nudge_y = ifelse(panel_title == "Traffic" & architecture == "GRU", 0.035, label_nudge_y),
    label_nudge_x = ifelse(panel_title == "Traffic" & architecture == "DLinear", 0.11, label_nudge_x),
    label_nudge_y = ifelse(panel_title == "Traffic" & architecture == "DLinear", 0.035, label_nudge_y)
  )

frontier_df <- bind_rows(
  baseline_points,
  improved_points
) %>%
  frontier_by_dataset("MSE_test", "D_w") %>%
  mutate(panel_title = factor(panel_title, levels = panel_levels))

plot <- ggplot() +
  geom_path(
    data = frontier_df,
    aes(x = MSE_test, y = D_w, group = panel_title, linetype = "Pareto frontier"),
    color = "#8a8a8a",
    linewidth = 0.7
  ) +
  geom_segment(
    data = subset(arrow_df, line_only),
    aes(
      x = arrow_xstart,
      y = arrow_ystart,
      xend = arrow_xend,
      yend = arrow_yend,
      color = architecture_family
    ),
    linewidth = 0.85,
    alpha = 0.8,
    lineend = "round",
    show.legend = FALSE
  ) +
  geom_segment(
    data = subset(arrow_df, !line_only),
    aes(
      x = arrow_xstart,
      y = arrow_ystart,
      xend = arrow_xend,
      yend = arrow_yend,
      color = architecture_family
    ),
    arrow = grid::arrow(length = grid::unit(0.18, "cm"), type = "closed"),
    linewidth = 0.85,
    alpha = 0.8,
    lineend = "round",
    show.legend = FALSE
  ) +
  geom_point(
    data = baseline_points,
    aes(
      x = MSE_test,
      y = D_w,
      color = architecture_family,
      shape = baseline_legend_label
    ),
    size = 3.2,
    stroke = 0.8
  ) +
  geom_point(
    data = improved_points %>%
      mutate(improvement_label = improvement_label_value),
    aes(
      x = MSE_test,
      y = D_w,
      color = architecture_family,
      shape = improvement_label
    ),
    size = 3.3,
    stroke = 1.1,
    fill = "white"
  ) +
  ggrepel::geom_text_repel(
    data = label_points,
    aes(
      x = MSE_test,
      y = D_w,
      label = architecture,
      color = architecture_family
    ),
    seed = 42,
    nudge_x = label_points$label_nudge_x,
    nudge_y = label_points$label_nudge_y,
    family = "serif",
    size = 2.8,
    box.padding = 0.16,
    point.padding = 0.18,
    min.segment.length = 0,
    max.overlaps = Inf,
    segment.alpha = 0.20,
    segment.size = 0.2,
    show.legend = FALSE
  ) +
  facet_wrap(~panel_title, nrow = 1, scales = "free_x") +
  ggh4x::facetted_pos_scales(
    x = list(
      panel_title == "Beijing Air Tiantan" ~ scale_x_continuous(
        limits = c(0.276, 0.364),
        breaks = c(0.28, 0.30, 0.32, 0.34, 0.36),
        labels = scales::label_number(accuracy = 0.01, drop0trailing = TRUE),
        expand = expansion(mult = c(0, 0))
      ),
      panel_title == "Penmanshiel WT08" ~ scale_x_continuous(
        limits = c(0.3335, 0.3565),
        breaks = c(0.335, 0.340, 0.345, 0.350, 0.355),
        labels = scales::label_number(accuracy = 0.001, drop0trailing = TRUE),
        expand = expansion(mult = c(0, 0))
      ),
      panel_title == "ETTh1" ~ scale_x_continuous(
        limits = c(0.41, 0.69),
        breaks = c(0.42, 0.48, 0.54, 0.60, 0.66),
        labels = scales::label_number(accuracy = 0.01, drop0trailing = TRUE),
        expand = expansion(mult = c(0, 0))
      ),
      panel_title == "Traffic" ~ scale_x_continuous(
        limits = c(0.41, 0.77),
        breaks = c(0.42, 0.5, 0.58, 0.66, 0.74),
        labels = scales::label_number(accuracy = 0.01, drop0trailing = TRUE),
        expand = expansion(mult = c(0, 0))
      )
    )
  ) +
  scale_color_manual(values = backbone_family_palette, drop = FALSE) +
  scale_shape_manual(
    values = shape_values,
    breaks = c(baseline_legend_label, improvement_label_value)
  ) +
  scale_linetype_manual(values = c("Pareto frontier" = "22")) +
  scale_y_continuous(
    expand = expansion(mult = c(0.05, 0.10)),
    breaks = c(seq(1, 1.8, 0.2))
  ) +
  labs(
    x = "$\\mathrm{MSE}_c$", 
    y = "$\\mathcal{D}_w$"
  ) +
  paper_theme() +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.997, 0.992),
    legend.justification = c(1, 1),
    legend.box = "vertical",
    legend.title = element_blank(),
    legend.text = element_text(size = 7.5),
    legend.key = element_blank(),
    legend.key.height = grid::unit(0.34, "cm"),
    legend.key.width = grid::unit(0.55, "cm"),
    legend.spacing.y = grid::unit(0.02, "cm"),
    legend.background = element_blank(),
    legend.box.background = element_rect(
      fill = "#FFFFFFE8",
      color = "#d9e2ef",
      linewidth = 0.4
    ),
    legend.box.margin = margin(0, 0, 0, 0),
    legend.margin = margin(1.5, 1.5, 1.5, 1.5)
  ) +
  guides(
    color = "none",
    shape = guide_legend(
      order = 1,
      override.aes = list(
        color = c("black", "#7f7f7f"),
        fill = c(NA, "white"),
        stroke = c(0.8, 1.1),
        size = c(2.4, 2.6)
      )
    ),
    linetype = guide_legend(
      order = 2,
      override.aes = list(color = "#8a8a8a", linewidth = 0.55)
    )
  )

save_plot(plot, output_path, width = 8.25, height = 3.25)
