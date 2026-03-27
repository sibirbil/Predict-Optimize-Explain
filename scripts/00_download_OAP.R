#!/usr/bin/env Rscript
# =============================================================================
# POE Research — Build “Wide Signals + CRSP” Panel (ML-ready)
#
# This script is adapted from Open Source Asset Pricing (OpenAP):
#   https://github.com/OpenSourceAP/CrossSectionDemos/blob/main/dl_signals_add_crsp.R
#
# Reference:
#   Chen, Andrew Y. and Tom Zimmermann (2022),
#   “Open Source Cross-Sectional Asset Pricing,” Critical Finance Review, 27(2), 207–264.
#   BibTeX:
#     @article{ChenZimmermann2022,
#       title={Open Source Cross-Sectional Asset Pricing},
#       author={Chen, Andrew Y. and Tom Zimmermann},
#       journal={Critical Finance Review},
#       year={2022},
#       volume={27},
#       number={2},
#       pages={207--264}
#     }
#
# POE-specific modifications vs OpenAP demo:
#   (1) Keep and export ret_tplus1 (t+1 CRSP return target; percent units)
#   (2) Save Parquet output (in addition to CSV) 
#
# Outputs:
#   - {outpath}/signed_predictors_all_wide.csv
#   - {outpath}/signed_predictors_all_wide.parquet
#
# =============================================================================

rm(list = ls())
tic <- Sys.time()

# -----------------------------------------------------------------------------
# 0) Libraries
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(tidyverse)
  library(googledrive)
  library(data.table)
  library(getPass)
  library(RPostgres)
})

# -----------------------------------------------------------------------------
# 1) User config
# -----------------------------------------------------------------------------
# OpenAP release folder (Aug 2025.10)
pathRelease <- "https://drive.google.com/drive/folders/1qQDuTsnyvWfEJR6nPBQZ8xxlq6bkLG_y"

# Output directory (set this to match your Python RAW_DATA_DIR if desired)
outpath <- "temp/"

# -----------------------------------------------------------------------------
# 2) Setup
# -----------------------------------------------------------------------------
dir.create(outpath, recursive = TRUE, showWarnings = FALSE)

# Trigger Google Drive auth (interactive)
pathRelease %>% drive_ls()

# WRDS login
user <- getPass("wrds username: ")
pass <- getPass("wrds password: ")

wrds <- dbConnect(
  Postgres(),
  host     = "wrds-pgdata.wharton.upenn.edu",
  port     = 9737,
  dbname   = "wrds",
  user     = user,
  password = pass,
  sslmode  = "require"
)

# -----------------------------------------------------------------------------
# 3) Pull CRSP monthly + delisting info from WRDS
# -----------------------------------------------------------------------------
crspm <- dbSendQuery(
  conn = wrds,
  statement = "
    select a.permno, a.date, a.ret, a.shrout, a.prc,
           b.exchcd,
           c.dlstcd, c.dlret
    from crsp.msf as a
    left join crsp.msenames as b
      on a.permno=b.permno
     and b.namedt<=a.date
     and a.date<=b.nameendt
    left join crsp.msedelist as c
      on a.permno=c.permno
     and date_trunc('month', a.date) = date_trunc('month', c.dlstdt)
  "
) %>%
  dbFetch(n = -1) %>%
  setDT()

# -----------------------------------------------------------------------------
# 4) Delisting adjustment + formatting
# -----------------------------------------------------------------------------
crspm2 <- crspm %>%
  mutate(
    # Fill missing dlret using standard conventions (OpenAP demo logic)
    dlret = ifelse(
      is.na(dlret) &
        (dlstcd == 500 | (dlstcd >= 520 & dlstcd <= 584)) &
        (exchcd == 1 | exchcd == 2),
      -0.35,
      dlret
    ),
    dlret = ifelse(
      is.na(dlret) &
        (dlstcd == 500 | (dlstcd >= 520 & dlstcd <= 584)) &
        (exchcd == 3),
      -0.55,
      dlret
    ),
    dlret = ifelse(!is.na(dlret) & dlret < -1, -1, dlret),
    dlret = ifelse(is.na(dlret), 0, dlret),

    # Delisting-adjusted return
    ret = (1 + ret) * (1 + dlret) - 1,
    ret = ifelse(is.na(ret) & (dlret != 0), dlret, ret)
  ) %>%
  mutate(
    # Convert to percent + compute market cap + YYYYMM
    ret    = 100 * ret,
    date   = as.Date(date),
    me     = abs(prc) * shrout,
    yyyymm = lubridate::year(date) * 100 + lubridate::month(date)
  )

# -----------------------------------------------------------------------------
# 5) POE modification (1): Create t+1 return target ret_tplus1
# -----------------------------------------------------------------------------
crspm2 <- crspm2 %>%
  arrange(permno, yyyymm) %>%
  group_by(permno) %>%
  mutate(ret_tplus1 = lead(ret)) %>%   # target: next-month (percent) return
  ungroup()

# Signed CRSP predictors + target
crspmsignal <- crspm2 %>%
  transmute(
    permno,
    yyyymm,
    ret_tplus1,                                     # target variable (POE modification)
    STreversal = -1 * if_else(is.na(ret), 0, ret),   # short-term reversal (uses ret(t))
    Price      = -1 * log(abs(prc)),
    Size       = -1 * log(me)
  )

# -----------------------------------------------------------------------------
# 6) Download OpenAP wide signed predictors zip and read
# -----------------------------------------------------------------------------
target_dribble <- pathRelease %>% drive_ls() %>%
  filter(name == "Firm Level Characteristics") %>% drive_ls() %>%
  filter(name == "Full Sets") %>% drive_ls() %>%
  filter(name == "signed_predictors_dl_wide.zip")

zip_path <- file.path(outpath, "deleteme.zip")
drive_download(target_dribble, path = zip_path, overwrite = TRUE)

unzip(zip_path, exdir = gsub("/$", "", outpath))
wide_dl_raw <- fread(file.path(outpath, "signed_predictors_dl_wide.csv"))
file.remove(file.path(outpath, "signed_predictors_dl_wide.csv"))

# -----------------------------------------------------------------------------
# 7) Merge and export
# -----------------------------------------------------------------------------
signalwide <- full_join(
  wide_dl_raw,
  crspmsignal,
  by = c("permno", "yyyymm")
)

out_csv <- file.path(outpath, "signed_predictors_all_wide.csv")
fwrite(signalwide, file = out_csv, row.names = FALSE)

# POE modification (2): Save Parquet if {arrow} is available
if (requireNamespace("arrow", quietly = TRUE)) {
  out_pq <- file.path(outpath, "signed_predictors_all_wide.parquet")
  arrow::write_parquet(signalwide, sink = out_pq, compression = "snappy")
} else {
  message("Note: Package 'arrow' not installed; skipping Parquet output.")
  message("      Install with: install.packages('arrow')  (or your preferred method)")
}

gc()

# -----------------------------------------------------------------------------
# 8) Summaries (coverage by month)
# -----------------------------------------------------------------------------
obs <- as_tibble(signalwide) %>%
  select(-permno) %>%
  group_by(yyyymm) %>%
  summarize(across(everything(), ~ sum(!is.na(.x))), .groups = "drop")

widesum <- obs %>%
  pivot_longer(
    cols = -yyyymm,
    names_to = "signalname",
    values_to = "obs"
  ) %>%
  filter(obs >= 1) %>%
  group_by(signalname) %>%
  summarize(
    date_begin = min(yyyymm),
    date_end   = max(yyyymm),
    mean_firmobs_per_month = floor(mean(obs)),
    .groups = "drop"
  ) %>%
  as.data.frame()

cat(sprintf("\nSaved: %s\n", out_csv))
cat("Top signals by coverage (first 10):\n")
setDT(widesum)
print(widesum, topn = 10)

datelist <- sort(unique(signalwide$yyyymm), decreasing = TRUE)
datelist <- datelist[1:min(24, length(datelist))]

recent_nobs <- as_tibble(signalwide) %>%
  select(-permno) %>%
  filter(yyyymm %in% datelist) %>%
  group_by(yyyymm) %>%
  summarize(across(everything(), ~ sum(!is.na(.x))), .groups = "drop") %>%
  t()

cat("\nNumber of firms with data in recent months:\n")
print(recent_nobs[1:10, 1:12])
cat("...\n")
print(recent_nobs[c(1, nrow(recent_nobs) - 0:10), 1:12])

# -----------------------------------------------------------------------------
# 9) Cleanup
# -----------------------------------------------------------------------------
dbDisconnect(wrds)

toc <- Sys.time()
cat(sprintf("\nDone. Runtime: %.2f minutes\n", as.numeric(difftime(toc, tic, units = "mins"))))
cat(sprintf("Output directory: %s\n", normalizePath(outpath, winslash = "/", mustWork = FALSE)))
