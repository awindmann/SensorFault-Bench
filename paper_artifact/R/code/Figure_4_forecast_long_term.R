script_dir <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  script_arg <- cmd_args[startsWith(cmd_args, file_arg)]

  if (length(script_arg) > 0) {
    return(dirname(normalizePath(sub(file_arg, "", script_arg[1]), mustWork = TRUE)))
  }

  for (frame in rev(sys.frames())) {
    if (!is.null(frame$ofile) && nzchar(frame$ofile)) {
      return(dirname(normalizePath(frame$ofile, mustWork = TRUE)))
    }
  }

  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    path <- rstudioapi::getActiveDocumentContext()$path
    if (nzchar(path)) {
      return(dirname(normalizePath(path, mustWork = TRUE)))
    }
  }

  getwd()
}

script_path_dir <- script_dir()
figure_root <- if (basename(script_path_dir) == "code") {
  dirname(script_path_dir)
} else {
  script_path_dir
}
data_dir <- file.path(figure_root, "data")
figures_dir <- file.path(figure_root, "figures")
local_r_lib <- file.path(figure_root, "library")
if (dir.exists(local_r_lib)) {
  .libPaths(c(local_r_lib, .libPaths()))
}

required_packages <- c("ggplot2", "ggrepel", "tikzDevice")
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
  library(ggrepel)
  library(tikzDevice)
})

trace_root <- file.path(data_dir, "forecast_plots_final_traces")

compile_tikz_pdf <- function(tex_path) {
  old_wd <- getwd()
  tex_dir <- dirname(tex_path)
  tex_file <- basename(tex_path)
  pdf_path <- sub("\\.tex$", ".pdf", tex_path)
  aux_path <- sub("\\.tex$", ".aux", tex_path)
  log_path <- sub("\\.tex$", ".log", tex_path)

  on.exit(setwd(old_wd), add = TRUE)
  setwd(tex_dir)

  tex_cache_dir <- file.path(tempdir(), "luatex-cache")
  dir.create(tex_cache_dir, recursive = TRUE, showWarnings = FALSE)
  old_texmfvar <- Sys.getenv("TEXMFVAR", unset = NA_character_)
  old_texmfcache <- Sys.getenv("TEXMFCACHE", unset = NA_character_)
  on.exit({
    if (is.na(old_texmfvar)) {
      Sys.unsetenv("TEXMFVAR")
    } else {
      Sys.setenv(TEXMFVAR = old_texmfvar)
    }
    if (is.na(old_texmfcache)) {
      Sys.unsetenv("TEXMFCACHE")
    } else {
      Sys.setenv(TEXMFCACHE = old_texmfcache)
    }
  }, add = TRUE)
  Sys.setenv(TEXMFVAR = tex_cache_dir, TEXMFCACHE = tex_cache_dir)

  compile_output <- system2(
    "lualatex",
    c("-interaction=nonstopmode", "-halt-on-error", tex_file),
    stdout = TRUE,
    stderr = TRUE
  )
  status <- attr(compile_output, "status")
  if (!is.null(status) && status != 0) {
    stop(
      "lualatex failed for ",
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

theme_set(
  theme_bw() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(fill = NA, color = "#333333", linewidth = 0.65),
      strip.background = element_rect(fill = "white", color = "#333333", linewidth = 0.65),
      axis.text = element_text(size = 12, color = "black"),
      axis.title = element_text(size = 14, color = "black", face = "bold"),
      axis.ticks = element_line(color = "black", linewidth = 0.35),
      legend.title = element_blank()
    )
)

cases <- data.frame(
  dataset_dir = c("etth1", "etth1", "traffic", "traffic"),
  dataset_label = c("ETTh1", "ETTh1", "Traffic", "Traffic"),
  model = c("PatchTST", "PatchTST", "PatchTST", "PatchTST"),
  scenario = c("stuck_sensor", "time_compress", "missing_data", "time_stretch"),
  scenario_label = c("StuckSensor", "TimeCompress", "MissingData", "TimeStretch"),
  file = c(
    "PatchTST_stuck_sensor_sid8111_sev0.186.csv",
    "PatchTST_time_compress_sid7968_sev0.495.csv",
    "PatchTST_missing_data_sid664_sev0.491.csv",
    "PatchTST_time_stretch_sid7218_sev0.494.csv"
  ),
  stringsAsFactors = FALSE
)

mcb <- c("#D55E00", "#56B4E9", "#E69F00", "#009E73", "#CC79A7")
max_focus_features <- 3
label_x_padding <- 6
show_other_channels <- TRUE

# Hand-tune each shown training label here after previewing the PDF.
training_label_adjustments <- data.frame(
  panel = c(1, 2, 2, 3, 3, 3, 4, 4, 4),
  label = c(
    "MUFL",
    "HUFL", "MUFL",
    "ch. 840", "ch. 701", "ch. 364",
    "ch. 840", "ch. 632", "ch. 586"
  ),
  x_offset = c(.5, 19.8, -9.0, 2.2, -.6, 0.15, -14, -13.4, 10),
  y_offset = c(1, 35.6, -35.1, 0.02, -0.055, 0.01, .46, 0.42, 0.3),
  stringsAsFactors = FALSE
)

# Hand-tune each shown forecast label here after previewing the PDF.
forecast_label_adjustments <- data.frame(
  panel = c(1, 2, 2, 3, 3, 3, 4, 4, 4),
  label = c(
    "MUFL",
    "HUFL", "MUFL",
    "ch. 840", "ch. 701", "ch. 364",
    "ch. 840", "ch. 632", "ch. 586"
  ),
  x_offset = c(-4, 1, 1, 2.2, 1.2, 1.2, 1, 1, 1.2),
  y_offset = c(0, 1, -1, 0.01, 0, 0, 0, 0, 0),
  stringsAsFactors = FALSE
)

# Hand-tune the phase headings shown in the top panel here.
phase_label_adjustments <- data.frame(
  panel = c(1, 1),
  label = c("input history", "forecast"),
  timestep = c(48.25, 144.25),
  y_offset = c(0, 0),
  stringsAsFactors = FALSE
)

is_true <- function(x) {
  x %in% c(TRUE, "True", "true", "TRUE", 1, "1")
}

changed_values <- function(clean, perturbed, tol = 1e-8) {
  one_missing <- xor(is.na(clean), is.na(perturbed))
  both_present <- !is.na(clean) & !is.na(perturbed)
  one_missing | (both_present & abs(perturbed - clean) > tol)
}

pretty_name <- function(x) {
  pretty <- gsub("_", " ", x)
  paste(toupper(substring(pretty, 1, 1)), substring(pretty, 2), sep = "")
}

short_feature_label <- function(x) {
  if (grepl("^[0-9]+$", x)) {
    return(paste0("ch. ", x))
  }
  if (nchar(x) > 22) {
    return(paste0(substr(x, 1, 19), "..."))
  }
  x
}

apply_label_adjustments <- function(dat, adjustments) {
  if (nrow(dat) == 0) {
    return(dat)
  }

  dat$x_offset <- 0
  dat$y_offset <- 0

  if (nrow(adjustments) == 0) {
    return(dat)
  }

  key <- paste(dat$panel, dat$label)
  adj_key <- paste(adjustments$panel, adjustments$label)
  match_idx <- match(key, adj_key)
  hit <- !is.na(match_idx)

  dat$x_offset[hit] <- adjustments$x_offset[match_idx[hit]]
  dat$y_offset[hit] <- adjustments$y_offset[match_idx[hit]]
  dat
}

finite_range <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) {
    return(c(0, 1))
  }
  range(x)
}

axis_breaks <- function(max_x, split_x) {
  raw_breaks <- c(1, 24, 48, 72, split_x - 0.5, 120, 144, 168, max_x)
  unique(raw_breaks[raw_breaks >= 1 & raw_breaks <= max_x])
}

long_rows <- function(dat, case_id, vars, segment, value_col, series, focus) {
  rows <- list()

  for (var in vars) {
    part <- dat[dat$segment == segment & dat$feature == var, ]
    if (nrow(part) == 0) {
      next
    }
    rows[[length(rows) + 1]] <- data.frame(
      timestep = part$plot_timestep,
      value = part[[value_col]],
      variable = var,
      series = series,
      focus = focus,
      trace = paste(segment, value_col, sep = "_"),
      panel = case_id,
      stringsAsFactors = FALSE
    )
  }

  if (length(rows) == 0) {
    return(NULL)
  }
  do.call(rbind, rows)
}

input_focus_rows <- function(dat, case_id, vars) {
  rows <- list()

  for (var in vars) {
    part <- dat[dat$segment == "input" & dat$feature == var, ]
    if (nrow(part) == 0) {
      next
    }
    changed <- changed_values(part$clean_input, part$perturbed_input)

    if (!any(changed, na.rm = TRUE)) {
      next
    }

    changed_range <- range(which(changed))
    changed_interval <- seq.int(changed_range[1], changed_range[2])
    display_range <- c(
      max(1, changed_range[1] - 1),
      min(nrow(part), changed_range[2] + 1)
    )
    display_interval <- seq.int(display_range[1], display_range[2])

    normal_value <- part$perturbed_input
    normal_value[changed_interval] <- NA_real_

    fault_value <- rep(NA_real_, nrow(part))
    fault_value[display_interval] <- part$perturbed_input[display_interval]

    clean_reference <- rep(NA_real_, nrow(part))
    clean_reference[display_interval] <- part$clean_input[display_interval]

    rows[[length(rows) + 1]] <- data.frame(
      timestep = part$plot_timestep,
      value = normal_value,
      variable = var,
      series = "Affected sensor",
      focus = "focus",
      trace = "affected_sensor",
      panel = case_id,
      stringsAsFactors = FALSE
    )
    rows[[length(rows) + 1]] <- data.frame(
      timestep = part$plot_timestep,
      value = clean_reference,
      variable = var,
      series = "Clean interval reference",
      focus = "focus",
      trace = "clean_interval_reference",
      panel = case_id,
      stringsAsFactors = FALSE
    )
    rows[[length(rows) + 1]] <- data.frame(
      timestep = part$plot_timestep,
      value = fault_value,
      variable = var,
      series = "Faulted sensor interval",
      focus = "focus",
      trace = "faulted_sensor_interval",
      panel = case_id,
      stringsAsFactors = FALSE
    )
  }

  do.call(rbind, rows)
}

choose_focus_features <- function(dat, affected) {
  scores <- vapply(
    affected,
    function(var) {
      part <- dat[dat$segment == "input" & dat$feature == var, ]
      diff <- abs(part$perturbed_input - part$clean_input)
      if (all(!is.finite(diff))) {
        return(0)
      }
      max(diff, na.rm = TRUE)
    },
    numeric(1)
  )
  affected[order(scores, decreasing = TRUE)][seq_len(min(max_focus_features, length(affected)))]
}

case_rows <- list()
input_labels <- list()
forecast_labels <- list()
split_rows <- list()
panel_label_text <- character(nrow(cases))

for (i in seq_len(nrow(cases))) {
  path <- file.path(trace_root, cases$dataset_dir[i], "traces", cases$scenario[i], cases$file[i])
  dat <- read.csv(path, check.names = FALSE)

  input_steps <- sort(unique(dat$time_step[dat$segment == "input"]))
  forecast_steps <- sort(unique(dat$time_step[dat$segment == "forecast"]))
  n_input <- length(input_steps)
  n_forecast <- length(forecast_steps)
  dat$plot_timestep <- ifelse(dat$segment == "input", dat$time_step + 1, n_input + dat$time_step + 1)

  vars <- unique(as.character(dat$feature))
  affected <- unique(as.character(dat$feature[is_true(dat$affected_feature)]))
  focus_vars <- choose_focus_features(dat, affected)
  context_vars <- setdiff(vars, focus_vars)
  split_rows[[length(split_rows) + 1]] <- data.frame(panel = i, split_x = n_input + 0.5)

  panel_label_text[i] <- paste0(
    cases$dataset_label[i],
    ", ",
    cases$model[i],
    ", ",
    cases$scenario_label[i],
    ", Severity = ",
    sprintf("%.3f", unique(dat$severity)[1])
  )

  if (length(focus_vars) > 0) {
    label_fractions <- seq(0.42, 0.62, length.out = length(focus_vars))

    for (j in seq_along(focus_vars)) {
      var <- focus_vars[j]
      input_part <- dat[dat$segment == "input" & dat$feature == var, ]
      changed <- changed_values(input_part$clean_input, input_part$perturbed_input)
      changed_idx <- which(changed)
      if (length(changed_idx) > 0) {
        label_idx <- changed_idx[
          max(1, min(length(changed_idx), round(length(changed_idx) * label_fractions[j])))
        ]
        input_y <- input_part$perturbed_input[label_idx]
        if (!is.finite(input_y)) {
          input_y <- input_part$clean_input[label_idx]
        }
        input_labels[[length(input_labels) + 1]] <- data.frame(
          timestep = input_part$plot_timestep[label_idx],
          value = input_y,
          label = short_feature_label(var),
          panel = i,
          stringsAsFactors = FALSE
        )
      }

      forecast_part <- dat[dat$segment == "forecast" & dat$feature == var, ]
      if (nrow(forecast_part) > 0) {
        forecast_labels[[length(forecast_labels) + 1]] <- data.frame(
          timestep = max(forecast_part$plot_timestep),
          value = forecast_part$perturbed_prediction[nrow(forecast_part)],
          label = short_feature_label(var),
          panel = i,
          stringsAsFactors = FALSE
        )
      }
    }
  }

  if (length(context_vars) > 0) {
    if (show_other_channels) {
      case_rows[[length(case_rows) + 1]] <- long_rows(dat, i, context_vars, "input", "clean_input", "Other sensors", "context")
      case_rows[[length(case_rows) + 1]] <- long_rows(dat, i, context_vars, "forecast", "ground_truth", "Other forecast traces", "context")
      case_rows[[length(case_rows) + 1]] <- long_rows(dat, i, context_vars, "forecast", "clean_prediction", "Other forecast traces", "context")
      case_rows[[length(case_rows) + 1]] <- long_rows(dat, i, context_vars, "forecast", "perturbed_prediction", "Other forecast traces", "context")
    }
  }

  case_rows[[length(case_rows) + 1]] <- input_focus_rows(dat, i, focus_vars)
  case_rows[[length(case_rows) + 1]] <- long_rows(dat, i, focus_vars, "forecast", "ground_truth", "Ground truth", "focus")
  case_rows[[length(case_rows) + 1]] <- long_rows(dat, i, focus_vars, "forecast", "clean_prediction", "Prediction from clean input", "focus")
  case_rows[[length(case_rows) + 1]] <- long_rows(dat, i, focus_vars, "forecast", "perturbed_prediction", "Prediction from faulted input", "focus")
}

plot_data <- do.call(rbind, case_rows)
legend_levels <- c(
  "Affected channels",
  "Clean interval reference",
  "Faulted sensor interval",
  if (show_other_channels) "Other channels",
  "Clean-input forecast",
  "Perturbed-input forecast"
)
plot_data$series <- factor(
  plot_data$series,
  levels = c(
    "Affected sensor",
    "Clean interval reference",
    "Faulted sensor interval",
    "Other sensors",
    "Ground truth",
    "Prediction from clean input",
    "Prediction from faulted input",
    "Other forecast traces"
  )
)
plot_data$legend_key <- factor(
  ifelse(
    plot_data$series %in% c("Affected sensor", "Ground truth"),
    "Affected channels",
    ifelse(
      plot_data$series %in% c("Other sensors", "Other forecast traces"),
      "Other channels",
      ifelse(
        plot_data$series == "Prediction from clean input",
        "Clean-input forecast",
        ifelse(
          plot_data$series == "Prediction from faulted input",
          "Perturbed-input forecast",
          as.character(plot_data$series)
        )
      )
    )
  ),
  levels = legend_levels
)
plot_data$panel <- factor(plot_data$panel, levels = seq_len(nrow(cases)))
plot_data$line_id <- interaction(plot_data$panel, plot_data$series, plot_data$trace, plot_data$variable, drop = TRUE)
context_plot_data <- plot_data[plot_data$focus == "context", ]
focus_plot_data <- plot_data[plot_data$focus == "focus", ]

input_label_data <- do.call(rbind, input_labels)
forecast_label_data <- do.call(rbind, forecast_labels)
if (!is.null(input_label_data)) {
  input_label_data$panel <- factor(input_label_data$panel, levels = seq_len(nrow(cases)))
}
if (!is.null(forecast_label_data)) {
  forecast_label_data$panel <- factor(forecast_label_data$panel, levels = seq_len(nrow(cases)))
}
input_label_data <- apply_label_adjustments(input_label_data, training_label_adjustments)
forecast_label_data <- apply_label_adjustments(forecast_label_data, forecast_label_adjustments)
manual_training_labels <- input_label_data[
  input_label_data$panel == 2 & input_label_data$label %in% c("HUFL", "MUFL"),
]
manual_training_labels$label_x <- manual_training_labels$timestep + manual_training_labels$x_offset
manual_training_labels$label_y <- manual_training_labels$value + manual_training_labels$y_offset
repel_input_label_data <- input_label_data[
  !(input_label_data$panel == 2 & input_label_data$label %in% c("HUFL", "MUFL")),
]
anchored_input_label_data <- repel_input_label_data[repel_input_label_data$panel == 4, ]
repel_input_label_data <- repel_input_label_data[repel_input_label_data$panel != 4, ]
anchored_forecast_label_data <- forecast_label_data[forecast_label_data$panel == 4, ]
repel_forecast_label_data <- forecast_label_data[forecast_label_data$panel != 4, ]
split_data <- do.call(rbind, split_rows)
split_data$panel <- factor(split_data$panel, levels = seq_len(nrow(cases)))

first_panel_values <- plot_data$value[plot_data$panel == levels(plot_data$panel)[1]]
first_range <- finite_range(first_panel_values)
max_x <- max(plot_data$timestep, na.rm = TRUE)
split_x <- split_data$split_x[1]
phase_label_data <- data.frame(
  timestep = phase_label_adjustments$timestep,
  value = first_range[2] + diff(first_range) * 0.11 + phase_label_adjustments$y_offset,
  label = phase_label_adjustments$label,
  panel = factor(phase_label_adjustments$panel, levels = seq_len(nrow(cases))),
  stringsAsFactors = FALSE
)

panel_labels <- setNames(panel_label_text, seq_len(nrow(cases)))

line_colors <- c(
  "Affected channels" = "#4A4A4A",
  "Clean interval reference" = "#6F6F6F",
  "Faulted sensor interval" = mcb[1],
  "Other channels" = "#C6C6C6",
  "Clean-input forecast" = mcb[4],
  "Perturbed-input forecast" = mcb[2]
)

line_types <- c(
  "Affected channels" = "solid",
  "Clean interval reference" = "longdash",
  "Faulted sensor interval" = "solid",
  "Other channels" = "solid",
  "Clean-input forecast" = "longdash",
  "Perturbed-input forecast" = "solid"
)

line_widths <- c(
  "Affected channels" = 0.45,
  "Clean interval reference" = 0.46,
  "Faulted sensor interval" = 0.72,
  "Other channels" = 0.12,
  "Clean-input forecast" = 0.54,
  "Perturbed-input forecast" = 0.66
)

line_alphas <- c(
  "Affected channels" = 0.92,
  "Clean interval reference" = 0.95,
  "Faulted sensor interval" = 0.96,
  "Other channels" = 0.34,
  "Clean-input forecast" = 0.88,
  "Perturbed-input forecast" = 0.94
)

legend_labels <- c(
  "Affected channels" = "Affected channels",
  "Clean interval reference" = "Clean reference",
  "Faulted sensor interval" = "Fault interval",
  "Other channels" = "Other channels",
  "Clean-input forecast" = "Clean-input forecast",
  "Perturbed-input forecast" = "Perturbed-input forecast"
)

p <- ggplot(
  focus_plot_data,
  aes(timestep, value, color = legend_key, linetype = legend_key, linewidth = legend_key, alpha = legend_key, group = line_id)
) +
  geom_line(data = context_plot_data, na.rm = TRUE, lineend = "round") +
  geom_line(na.rm = TRUE, lineend = "round") +
  geom_vline(data = split_data, aes(xintercept = split_x), color = "#8A8A8A", linewidth = 0.55) +
  facet_wrap(~ panel, ncol = 1, labeller = as_labeller(panel_labels), scales = "free_y") +
  geom_text(data = phase_label_data, aes(timestep, value, label = label), inherit.aes = FALSE, size = 3.1, fontface = "bold", color = "#222222") +
  geom_text(
    data = manual_training_labels,
    aes(label_x, label_y, label = label),
    inherit.aes = FALSE,
    color = mcb[1],
    size = 2.5,
    fontface = "bold"
  ) +
  geom_text_repel(
    data = repel_input_label_data,
    aes(timestep, value, label = label),
    inherit.aes = FALSE,
    color = mcb[1],
    size = 2.5,
    fontface = "bold",
    box.padding = 0.34,
    point.padding = 0.55,
    nudge_x = repel_input_label_data$x_offset,
    nudge_y = repel_input_label_data$y_offset,
    min.segment.length = 0,
    segment.color = NA,
    force = 4,
    seed = 42,
    max.overlaps = Inf
  ) +
  geom_text_repel(
    data = anchored_input_label_data,
    aes(timestep, value, label = label),
    inherit.aes = FALSE,
    color = mcb[1],
    size = 2.5,
    fontface = "bold",
    box.padding = 0.34,
    point.padding = 0.55,
    nudge_x = anchored_input_label_data$x_offset,
    nudge_y = anchored_input_label_data$y_offset,
    min.segment.length = 0,
    segment.color = NA,
    force = 4,
    seed = 42,
    max.overlaps = Inf
  ) +
  geom_text_repel(
    data = repel_forecast_label_data,
    aes(timestep, value, label = label),
    inherit.aes = FALSE,
    color = mcb[2],
    size = 2.3,
    fontface = "bold",
    box.padding = 0.34,
    point.padding = 0.55,
    nudge_x = repel_forecast_label_data$x_offset,
    nudge_y = repel_forecast_label_data$y_offset,
    min.segment.length = 0,
    segment.color = NA,
    force = 4,
    seed = 42,
    max.overlaps = Inf
  ) +
  geom_text_repel(
    data = anchored_forecast_label_data,
    aes(timestep, value, label = label),
    inherit.aes = FALSE,
    color = mcb[2],
    size = 2.3,
    fontface = "bold",
    box.padding = 0.34,
    point.padding = 0.55,
    nudge_x = anchored_forecast_label_data$x_offset,
    nudge_y = anchored_forecast_label_data$y_offset,
    min.segment.length = 0,
    segment.color = NA,
    force = 4,
    seed = 42,
    max.overlaps = Inf
  ) +
  scale_color_manual(values = line_colors, breaks = legend_levels, labels = legend_labels[legend_levels], drop = FALSE) +
  scale_linetype_manual(values = line_types, drop = FALSE) +
  scale_linewidth_manual(values = line_widths, drop = FALSE) +
  scale_alpha_manual(values = line_alphas, drop = FALSE) +
  scale_x_continuous(breaks = axis_breaks(max_x, split_x), limits = c(1, max_x + label_x_padding)) +
  scale_y_continuous(expand = expansion(mult = c(0.08, 0.18))) +
  coord_cartesian(clip = "off") +
  labs(x = "Time [h]", y = "Sensor value", color = NULL, linewidth = NULL, alpha = NULL) +
  guides(
    color = guide_legend(
      ncol = 2,
      byrow = FALSE,
      override.aes = list(
        linewidth = unname(line_widths[legend_levels]),
        alpha = pmin(unname(line_alphas[legend_levels]) + 0.18, 1),
        linetype = unname(line_types[legend_levels])
      ),
      order = 1
    ),
    linetype = "none",
    linewidth = "none",
    alpha = "none"
  ) +
  theme(
    strip.text = element_text(color = "#111111", size = 8.2, face = "bold"),
    legend.position = "inside",
    legend.position.inside = c(0.012, 0.84),
    legend.justification = c(0, 1),
    legend.direction = "horizontal",
    legend.text = element_text(color = "#111111", size = 7.8),
    legend.key.height = unit(0.18, "cm"),
    legend.key.width = unit(0.54, "cm"),
    legend.margin = margin(2, 3, 2, 3),
    legend.spacing.x = unit(0.10, "cm"),
    legend.spacing.y = unit(0.08, "cm"),
    legend.background = element_blank(),
    legend.box.background = element_blank(),
    panel.spacing.y = unit(0.35, "cm"),
    plot.margin = margin(3, 4, 3, 4)
  )

save_plot(
  p,
  file.path(figures_dir, "Figure_4_forecast_long_term.pdf"),
  width = 8.0,
  height = 11.0
)
