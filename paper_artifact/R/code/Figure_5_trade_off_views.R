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

required_packages <- c("ggplot2", "dplyr", "tikzDevice", "ggh4x", "ggrepel")
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
  library(ggrepel)
})

out_dir <- figures_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

main_table_backbone_path <- file.path(data_dir, "main_table_backbone.csv")
if (!file.exists(main_table_backbone_path)) {
  stop("Missing local input file: ", main_table_backbone_path, call. = FALSE)
}

dataset_labels <- c(
  "Beijing Tiantan" = "Beijing Air Tiantan",
  "Penmanshiel WT08" = "Penmanshiel WT08",
  "ETTh1" = "ETTh1",
  "traffic" = "Traffic"
)

architecture_palette <- c(
  "SeasonalNaive" = "#0072B2",
  "DLinear" = "#D55E00",
  "GRU" = "#009E73",
  "ModernTCN" = "#E69F00",
  "TSMixer" = "#56B4E9",
  "PatchTST" = "#CC79A7",
  "Chronos2" = "#999999"
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

accent_color <- "#7F7F7F"
baseline_legend_label <- "Baseline model"

paper_theme <- function() {
  theme_bw(base_size = 10, base_family = "serif") +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      strip.background = element_blank(),
      axis.text.x = element_text(color="black", size = 9),
      axis.text.y = element_text(color="black", size = 9),
      strip.text = element_text(size = 10, face = "bold"),
      legend.position = "none",
      axis.title = element_text(size = 11),
      plot.margin = margin(0, 0, 0, 0)
    )
}

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

save_plot <- function(plot, filename, width, height) {
  pdf_path <- file.path(out_dir, filename)
  tex_path <- sub("\\.pdf$", ".tex", pdf_path)
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

reshape_metric_table <- function(df, id_cols, value_cols, value_name, key_name) {
  long_df <- reshape(
    df,
    varying = value_cols,
    v.names = value_name,
    timevar = key_name,
    times = value_cols,
    direction = "long"
  )
  row.names(long_df) <- NULL
  long_df
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
  parts <- split(df, df$dataset_label, drop = TRUE)
  bind_rows(lapply(parts, pareto_frontier, x_col = x_col, y_col = y_col))
}

extract_metric_block <- function(
  df,
  measure_col,
  value_cols,
  expected_measures,
  context,
  measure_map = NULL
) {
  if (length(value_cols) == 0) {
    stop(context, ": expected at least one value column.", call. = FALSE)
  }

  block_df <- df[, c("dataset", measure_col, value_cols), drop = FALSE]
  names(block_df)[names(block_df) == measure_col] <- "measure"

  if (!is.null(measure_map)) {
    unknown_measures <- setdiff(unique(block_df$measure), names(measure_map))
    if (length(unknown_measures) > 0) {
      stop(
        context,
        ": unsupported measure values ",
        paste(unknown_measures, collapse = ", "),
        call. = FALSE
      )
    }
    block_df$measure <- unname(measure_map[block_df$measure])
  }

  missing_measures <- setdiff(expected_measures, unique(block_df$measure))
  if (length(missing_measures) > 0) {
    stop(
      context,
      ": missing measures ",
      paste(missing_measures, collapse = ", "),
      call. = FALSE
    )
  }

  reshape_metric_table(
    block_df,
    id_cols = c("dataset", "measure"),
    value_cols = value_cols,
    value_name = "value",
    key_name = "key"
  )
}

clean_vs_limits <- function(df, x_col, y_col) {
  limits_matrix <- do.call(
    rbind,
    lapply(split(df, df$dataset_label, drop = TRUE), function(panel_df) {
      lower <- min(c(panel_df[[x_col]], panel_df[[y_col]]), na.rm = TRUE)
      upper <- max(c(panel_df[[x_col]], panel_df[[y_col]]), na.rm = TRUE)
      c(lower = lower, upper = upper)
    })
  )
  data.frame(
    dataset_label = rownames(limits_matrix),
    lower = limits_matrix[, "lower"],
    upper = limits_matrix[, "upper"],
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}

dataset_mse_c_scales <- list(
  dataset_label == "Beijing Air Tiantan" ~ scale_x_continuous(
    limits = c(0.28, 0.362),
    breaks = c(0.28, 0.30, 0.32, 0.34, 0.36)
  ),
  dataset_label == "Penmanshiel WT08" ~ scale_x_continuous(
    limits = c(0.33, 0.41),
    breaks = c(0.33, 0.35, 0.37, 0.39, 0.41)
  ),
  dataset_label == "ETTh1" ~ scale_x_continuous(
    expand = expansion(mult = c(0.05, 0.10)),
    breaks = scales::breaks_width(0.05)
  ),
  dataset_label == "Traffic" ~ scale_x_continuous(
    limits = c(0.4, 1.3),
    breaks = c(0.4, 0.6, 0.8, 1.0, 1.2)
  )
)

main_table_backbone_raw <- read.csv(
  main_table_backbone_path,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  na.strings = c("--", "NA", "")
)
require_columns(
  main_table_backbone_raw,
  c("dataset", "dataset_label", "backbone", "D_w", "MSE_c", "MSE_w"),
  basename(main_table_backbone_path)
)
unknown_models <- setdiff(unique(main_table_backbone_raw$backbone), names(architecture_palette))
if (length(unknown_models) > 0) {
  stop(
    basename(main_table_backbone_path),
    ": unsupported backbone values ",
    paste(unknown_models, collapse = ", "),
    call. = FALSE
  )
}

baseline_df <- main_table_backbone_raw %>%
  transmute(
    dataset = dataset,
    dataset_label = ifelse(dataset_label == "traffic", "Traffic", dataset_label),
    model = backbone,
    D_w = D_w,
    MSE_c = MSE_c,
    MSE_w = MSE_w
  )
require_columns(baseline_df, c("dataset", "model", "D_w", "MSE_c", "MSE_w"), "backbone metrics")
baseline_df <- baseline_df %>%
  mutate(
    dataset_label = factor(dataset_label, levels = unname(dataset_labels)),
    model = factor(model, levels = names(architecture_palette))
  ) %>%
  filter(!is.na(dataset_label), !is.na(model))

baseline_pareto_df <- baseline_df %>%
  filter(!is.na(MSE_c), !is.na(D_w)) %>%
  mutate(
    architecture_family = unname(backbone_family_map[as.character(model)])
  )
if (any(is.na(baseline_pareto_df$architecture_family))) {
  missing_family_models <- unique(as.character(baseline_pareto_df$model[is.na(baseline_pareto_df$architecture_family)]))
  stop(
    "Missing backbone family mapping for models: ",
    paste(missing_family_models, collapse = ", "),
    call. = FALSE
  )
}
baseline_pareto_df <- baseline_pareto_df %>%
  mutate(
    architecture_family = factor(
      architecture_family,
      levels = names(backbone_family_palette)
    )
  )
pareto_df <- frontier_by_dataset(baseline_pareto_df, "MSE_c", "D_w")
backbone_label_df <- baseline_pareto_df %>%
  mutate(
    label_nudge_x = 0,
    label_nudge_y = 0,
    label_nudge_y = ifelse(dataset_label == "Beijing Air Tiantan" & model == "GRU", 0.05, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "Beijing Air Tiantan" & model == "ModernTCN", 0.05, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "Beijing Air Tiantan" & model == "PatchTST", 0.05, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Beijing Air Tiantan" & model == "SeasonalNaive", -0.005, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Beijing Air Tiantan" & model == "SeasonalNaive", -0.06, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Beijing Air Tiantan" & model == "Chronos2", 0.006, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Beijing Air Tiantan" & model == "Chronos2", 0.06, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "Beijing Air Tiantan" & model == "TSMixer", -0.045, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "Beijing Air Tiantan" & model == "DLinear", 0.05, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Penmanshiel WT08" & model == "TSMixer", 0.006, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Penmanshiel WT08" & model == "TSMixer", 0.048, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Penmanshiel WT08" & model == "ModernTCN", 0.002, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Penmanshiel WT08" & model == "ModernTCN", 0.035, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "Penmanshiel WT08" & model == "SeasonalNaive", 0.045, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "Penmanshiel WT08" & model == "Chronos2", 0.05, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "Penmanshiel WT08" & model == "DLinear", 0.05, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Penmanshiel WT08" & model == "PatchTST", 0.013, label_nudge_x),
    label_nudge_x = ifelse(dataset_label == "Penmanshiel WT08" & model == "GRU", 0.006, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Penmanshiel WT08" & model == "GRU", 0.035, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "ETTh1" & model == "ModernTCN", 0.002, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "ETTh1" & model == "ModernTCN", 0.03, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "ETTh1" & model == "Chronos2", 0.05, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "ETTh1" & model == "SeasonalNaive", 0.05, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "ETTh1" & model == "GRU", 0.05, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "ETTh1" & model == "PatchTST", 0.031, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "ETTh1" & model == "PatchTST", 0.044, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "ETTh1" & model == "DLinear", 0.038, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "ETTh1" & model == "DLinear", 0.02, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "ETTh1" & model == "ModernTCN", 0.017, label_nudge_x),
    label_nudge_x = ifelse(dataset_label == "ETTh1" & model == "TSMixer", 0.035, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "ETTh1" & model == "TSMixer", 0.02, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Traffic" & model == "PatchTST", 0.15, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Traffic" & model == "SeasonalNaive", 0.05, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Traffic" & model == "ModernTCN", 0.24, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Traffic" & model == "ModernTCN", -0.035, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Traffic" & model == "DLinear", 0.13, label_nudge_x),
    label_nudge_x = ifelse(dataset_label == "Traffic" & model == "TSMixer", 0.045, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Traffic" & model == "TSMixer", 0.04, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Traffic" & model == "GRU", 0.05, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Traffic" & model == "Chronos2", 0.05, label_nudge_y)
  )

backbone_pareto_plot <- ggplot(
  baseline_pareto_df,
  aes(x = MSE_c, y = D_w, color = architecture_family)
) +
  geom_path(
    data = pareto_df,
    aes(
      x = MSE_c,
      y = D_w,
      group = dataset_label,
      linetype = "Pareto frontier"
    ),
    inherit.aes = FALSE,
    color = accent_color,
    linewidth = 0.5
  ) +
  geom_point(aes(shape = baseline_legend_label), size = 3.2) +
  ggrepel::geom_text_repel(
    data = backbone_label_df,
    aes(label = model),
    seed = 42,
    nudge_x = backbone_label_df$label_nudge_x,
    nudge_y = backbone_label_df$label_nudge_y,
    size = 2.5,
    family = "serif",
    box.padding = 0.2,
    point.padding = 0.40,
    min.segment.length = 0,
    max.overlaps = Inf,
    segment.color = accent_color,
    segment.size = 0.25,
    show.legend = FALSE
  ) +
  facet_wrap(~dataset_label, nrow = 1, scales = "free_x") +
  scale_color_manual(values = backbone_family_palette, drop = FALSE) +
  scale_shape_manual(values = c("Baseline model" = 16)) +
  scale_linetype_manual(values = c("Pareto frontier" = "22")) +
  scale_x_continuous(expand = expansion(mult = c(0.05, 0.01))) +
  ggh4x::facetted_pos_scales(x = dataset_mse_c_scales) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.10)), breaks=c(seq(1,2.0,.2))) +
  coord_cartesian(clip = "on") +
  labs(x = "$\\mathrm{MSE}_c$", y = "$\\mathcal{D}_w$") +
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
        color = "black",
        size = 2.4
      )
    ),
    linetype = guide_legend(
      order = 2,
      override.aes = list(
        color = accent_color,
        linewidth = 0.55
      )
    )
  )

save_plot(
  backbone_pareto_plot,
  "Figure_5_backbone_pareto_by_dataset.pdf",
  width = 8.25,
  height = 3.
)

clean_vs_df <- baseline_df %>%
  filter(!is.na(MSE_c), !is.na(MSE_w)) %>%
  mutate(
    architecture_family = unname(backbone_family_map[as.character(model)])
  )
if (any(is.na(clean_vs_df$architecture_family))) {
  missing_family_models <- unique(as.character(clean_vs_df$model[is.na(clean_vs_df$architecture_family)]))
  stop(
    "Missing backbone family mapping for models in clean-vs-worst plot: ",
    paste(missing_family_models, collapse = ", "),
    call. = FALSE
  )
}
clean_vs_df <- clean_vs_df %>%
  mutate(
    architecture_family = factor(
      architecture_family,
      levels = names(backbone_family_palette)
    )
  )
limits_df <- clean_vs_limits(clean_vs_df, "MSE_c", "MSE_w") %>%
  mutate(dataset_label = factor(dataset_label, levels = levels(clean_vs_df$dataset_label)))

clean_vs_label_df <- clean_vs_df %>%
  mutate(
    label_nudge_x = 0,
    label_nudge_y = 0,
    label_nudge_x = ifelse(dataset_label == "Beijing Air Tiantan" & model == "GRU", 0.01, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Beijing Air Tiantan" & model == "GRU", -0.015, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "Beijing Air Tiantan" & model == "ModernTCN", 0.04, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Beijing Air Tiantan" & model == "PatchTST", -0.05, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Beijing Air Tiantan" & model == "PatchTST", -0.05, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Beijing Air Tiantan" & model == "SeasonalNaive", 0.065, label_nudge_x),
    label_nudge_x = ifelse(dataset_label == "Beijing Air Tiantan" & model == "Chronos2", 0.04, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Beijing Air Tiantan" & model == "Chronos2", 0.013, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Beijing Air Tiantan" & model == "TSMixer", -0.04, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Beijing Air Tiantan" & model == "TSMixer", -0.012, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "Beijing Air Tiantan" & model == "DLinear", 0.015, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Penmanshiel WT08" & model == "TSMixer", 0.035, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Penmanshiel WT08" & model == "TSMixer", 0.015, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Penmanshiel WT08" & model == "ModernTCN", 0.002, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Penmanshiel WT08" & model == "ModernTCN", 0.011, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "Penmanshiel WT08" & model == "SeasonalNaive", 0.012, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Penmanshiel WT08" & model == "Chronos2", 0.002, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Penmanshiel WT08" & model == "Chronos2", 0.012, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "Penmanshiel WT08" & model == "DLinear", 0.038, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Penmanshiel WT08" & model == "PatchTST", 0.034, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Penmanshiel WT08" & model == "GRU", 0.011, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "ETTh1" & model == "ModernTCN", 0.017, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "ETTh1" & model == "ModernTCN", 0.03, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "ETTh1" & model == "Chronos2", 0.025, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "ETTh1" & model == "SeasonalNaive", 0.025, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "ETTh1" & model == "GRU", 0.025, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "ETTh1" & model == "PatchTST", 0.031, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "ETTh1" & model == "PatchTST", 0.044, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "ETTh1" & model == "DLinear", 0.038, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "ETTh1" & model == "DLinear", -0.02, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "ETTh1" & model == "TSMixer", 0.035, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "ETTh1" & model == "TSMixer", 0.02, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Traffic" & model == "PatchTST", 0.15, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Traffic" & model == "PatchTST", -0.15, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "Traffic" & model == "SeasonalNaive", 0.07, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Traffic" & model == "ModernTCN", 0.24, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Traffic" & model == "ModernTCN", -0.035, label_nudge_y),
    label_nudge_x = ifelse(dataset_label == "Traffic" & model == "DLinear", 0.3, label_nudge_x),
    label_nudge_x = ifelse(dataset_label == "Traffic" & model == "TSMixer", -0.045, label_nudge_x),
    label_nudge_y = ifelse(dataset_label == "Traffic" & model == "TSMixer", 0.06, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "Traffic" & model == "GRU", 0.07, label_nudge_y),
    label_nudge_y = ifelse(dataset_label == "Traffic" & model == "Chronos2", 0.07, label_nudge_y)
  )

clean_vs_worst_plot <- ggplot(
  clean_vs_df,
  aes(x = MSE_c, y = MSE_w, color = architecture_family)
) +
  geom_segment(
    data = limits_df,
    aes(x = lower, y = lower, xend = upper, yend = upper),
    inherit.aes = FALSE,
    color = accent_color,
    linewidth = 0.45,
    linetype = "22"
  ) +
  geom_point(size = 3.2) +
  ggrepel::geom_text_repel(
    data = clean_vs_label_df,
    aes(label = model),
    seed = 42,
    nudge_x = clean_vs_label_df$label_nudge_x,
    nudge_y = clean_vs_label_df$label_nudge_y,
    size = 2.5,
    family = "serif",
    box.padding = 0.2,
    point.padding = 0.40,
    min.segment.length = 0,
    max.overlaps = Inf,
    segment.color = accent_color,
    segment.size = 0.25,
    show.legend = FALSE
  ) +
  facet_wrap(~dataset_label, nrow = 1, scales = "free") +
  scale_color_manual(values = backbone_family_palette, drop = FALSE) +
  scale_x_continuous(expand = expansion(mult = c(0.05, 0.10))) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.10))) +
  coord_cartesian(clip = "on") +
  labs(x = "$\\mathrm{MSE}_c$", y = "$\\mathrm{MSE}_w$") +
  paper_theme()

save_plot(
  clean_vs_worst_plot,
  "Figure_5_clean_vs_worst_error_by_dataset.pdf",
  width = 8.25, height = 3.
)

message("Completed local figure export from ", basename(main_table_backbone_path), ".")
