use std::{
    alloc::{GlobalAlloc, Layout, System},
    collections::HashMap,
    hash::BuildHasherDefault,
    hint::black_box,
    io::{self, Write as _},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use ahash::AHasher;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use hashbrown::HashMap as HashbrownMap;
use pomap::PoMap;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout as RataLayout},
    style::{Color, Style},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Chart, Dataset, Paragraph},
};

// ---------------------------------------------------------------------------
// Tracking allocator
// ---------------------------------------------------------------------------

static NET_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

struct TrackingAlloc;

unsafe impl GlobalAlloc for TrackingAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            NET_ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        NET_ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            NET_ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            if new_size > layout.size() {
                NET_ALLOCATED.fetch_add(new_size - layout.size(), Ordering::Relaxed);
            } else {
                NET_ALLOCATED.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL_ALLOC: TrackingAlloc = TrackingAlloc;

// ---------------------------------------------------------------------------
// Bench types & helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "bench-string")]
type BenchType = String;
#[cfg(not(feature = "bench-string"))]
type BenchType = u64;

type BenchKey = BenchType;
type BenchValue = BenchType;

type BenchHasher = AHasher;
type BenchHasherBuilder = BuildHasherDefault<BenchHasher>;
type BenchPoMap = PoMap<BenchKey, BenchValue, BenchHasherBuilder>;
type BenchHashMap = HashMap<BenchKey, BenchValue, BenchHasherBuilder>;
type BenchHashbrownMap = HashbrownMap<BenchKey, BenchValue, BenchHasherBuilder>;

fn random_bench_item(rng: &mut StdRng) -> BenchType {
    #[cfg(feature = "bench-string")]
    {
        use rand::distr::Alphanumeric;
        (0..128).map(|_| rng.sample(Alphanumeric) as char).collect()
    }
    #[cfg(not(feature = "bench-string"))]
    {
        rng.random()
    }
}

fn random_items(seed: u64, count: usize) -> Vec<BenchType> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|_| random_bench_item(&mut rng)).collect()
}

const MAX_INSERT_SIZE: usize = 10_000_000;
const NUM_INTERMEDIATE_ROUNDS: usize = 5;

fn insert_target_sizes() -> Vec<usize> {
    let mut power_targets = Vec::new();
    let mut current = 10usize;
    while current < MAX_INSERT_SIZE {
        power_targets.push(current);
        current *= 10;
    }
    power_targets.push(MAX_INSERT_SIZE);
    let rounds = NUM_INTERMEDIATE_ROUNDS.max(2);
    let mut targets = Vec::new();
    for window in power_targets.windows(2) {
        let start = window[0];
        let end = window[1];
        for step in 0..rounds {
            let size = start + (end - start) * step / (rounds - 1);
            if targets.last().copied() != Some(size) {
                targets.push(size);
            }
        }
    }
    if targets.is_empty() {
        targets.push(MAX_INSERT_SIZE);
    }
    targets
}

// ---------------------------------------------------------------------------
// TUI drawing
// ---------------------------------------------------------------------------

const IMPL_COLORS: [(&str, Color); 3] = [
    ("pomap", Color::Green),
    ("std_hashmap", Color::Yellow),
    ("hashbrown", Color::Red),
];

fn draw_tui(
    terminal: &mut Terminal<CrosstermBackend<io::Stderr>>,
    chart_data: &HashMap<&str, Vec<(f64, f64)>>,
    status: &str,
) {
    let _ = terminal.draw(|f| {
        let chunks = RataLayout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(4), Constraint::Length(2)])
            .split(f.area());

        let x_labels: Vec<Span> = ["10", "100", "1K", "10K", "100K", "1M", "10M"]
            .iter()
            .map(|s| Span::raw(*s))
            .collect();

        let all_impl_data: Vec<(&str, &Vec<(f64, f64)>)> = IMPL_COLORS
            .iter()
            .filter_map(|(name, _)| chart_data.get(*name).map(|d| (*name, d)))
            .collect();

        let avg_at = |x: f64| -> f64 {
            let (sum, count) = all_impl_data
                .iter()
                .filter_map(|(_, d)| d.iter().find(|(dx, _)| *dx == x).map(|(_, y)| *y))
                .fold((0.0, 0usize), |(s, c), y| (s + y, c + 1));
            if count > 0 { sum / count as f64 } else { 1.0 }
        };

        let datasets: Vec<Dataset> = IMPL_COLORS
            .iter()
            .filter_map(|(impl_name, color)| {
                let data = chart_data.get(*impl_name)?;
                let ratio: Vec<(f64, f64)> =
                    data.iter().map(|&(x, y)| (x, y / avg_at(x))).collect();
                let ratio = Vec::leak(ratio);
                Some(
                    Dataset::default()
                        .marker(symbols::Marker::Braille)
                        .graph_type(if cfg!(feature = "dots") {
                            ratatui::widgets::GraphType::Scatter
                        } else {
                            ratatui::widgets::GraphType::Line
                        })
                        .style(Style::default().fg(*color))
                        .data(ratio),
                )
            })
            .collect();

        let all_ratios = || {
            IMPL_COLORS
                .iter()
                .filter_map(|(name, _)| chart_data.get(*name))
                .flat_map(|v| v.iter().map(|&(x, y)| y / avg_at(x)))
        };
        let max_r = all_ratios().fold(1.0f64, f64::max);
        let min_r = all_ratios().fold(f64::MAX, f64::min).min(1.0);
        let margin = (max_r - min_r).max(0.01) * 0.15;
        let y_lo = (min_r - margin).max(0.0);
        let y_hi = max_r + margin;

        let y_lo_s = format!("{:.2}x", y_lo);
        let y_mid_s = format!("{:.2}x", (y_lo + y_hi) / 2.0);
        let y_hi_s = format!("{:.2}x", y_hi);

        let chart = Chart::new(datasets)
            .block(Block::default().title("insert_hot").borders(Borders::ALL))
            .x_axis(
                Axis::default()
                    .title("map_size")
                    .bounds([1.0, 7.0])
                    .labels(x_labels),
            )
            .y_axis(
                Axis::default()
                    .title("vs avg")
                    .bounds([y_lo, y_hi])
                    .labels(vec![
                        Span::raw(y_lo_s),
                        Span::raw(y_mid_s),
                        Span::raw(y_hi_s),
                    ]),
            );

        f.render_widget(chart, chunks[0]);

        let bottom = RataLayout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Length(1)])
            .split(chunks[1]);

        let legend = Line::from(vec![
            Span::styled("■ pomap", Style::default().fg(Color::Green)),
            Span::raw("  "),
            Span::styled("■ std_hashmap", Style::default().fg(Color::Yellow)),
            Span::raw("  "),
            Span::styled("■ hashbrown", Style::default().fg(Color::Red)),
        ]);
        f.render_widget(Paragraph::new(legend), bottom[0]);
        f.render_widget(Paragraph::new(status), bottom[1]);
    });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let target_sizes = insert_target_sizes();
    let max_size = *target_sizes.iter().max().unwrap();

    let keys: Vec<BenchKey> = random_items(0xA11CE, max_size);
    let values: Vec<BenchValue> = random_items(0xFACE, max_size);

    const ROUNDS: u64 = 20;

    let mut file = std::fs::File::create("insert_hot_graph.csv")
        .expect("failed to create insert_hot_graph.csv");
    writeln!(file, "map_size,impl,per_insert_ns").unwrap();

    // TUI setup
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stderr(), LeaveAlternateScreen);
        original_hook(info);
    }));
    enable_raw_mode().unwrap();
    execute!(io::stderr(), EnterAlternateScreen).unwrap();
    let backend = CrosstermBackend::new(io::stderr());
    let mut terminal = Terminal::new(backend).unwrap();

    // impl_name -> Vec<(log10_size, per_insert_ns)>
    let mut chart_data: HashMap<&str, Vec<(f64, f64)>> = HashMap::new();

    // Timing macro: create one map, then repeatedly clear and re-insert `size` entries.
    // The map's backing allocation stays warm across rounds (no dealloc/realloc),
    // so this measures hot insert performance with the allocator already primed.
    macro_rules! time_inserts {
        ($size:expr, $impl_name:expr, $new_map:expr) => {{
            let size = $size;
            let mut map = $new_map;
            // Prime: insert once so the map grows to its final allocation.
            for idx in 0..size {
                map.insert(keys[idx].clone(), values[idx].clone());
            }
            map.clear();
            let mut total_ns = 0u128;
            for _ in 0..ROUNDS {
                let start = Instant::now();
                for idx in 0..size {
                    black_box(map.insert(keys[idx].clone(), values[idx].clone()));
                }
                total_ns += start.elapsed().as_nanos();
                map.clear();
            }
            let per_insert = total_ns as f64 / (size as f64 * ROUNDS as f64);
            writeln!(file, "{},{},{:.2}", size, $impl_name, per_insert).unwrap();
            let x = (size as f64).log10();
            chart_data
                .entry($impl_name)
                .or_insert_with(Vec::new)
                .push((x, per_insert));
        }};
    }

    // Background thread for quit on 'q' or Ctrl+C.
    let quit = Arc::new(AtomicBool::new(false));
    let quit_bg = quit.clone();
    let _input_thread = std::thread::spawn(move || {
        loop {
            if event::poll(Duration::from_millis(50)).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.code == KeyCode::Char('q')
                        || (key.code == KeyCode::Char('c')
                            && key
                                .modifiers
                                .contains(crossterm::event::KeyModifiers::CONTROL))
                    {
                        quit_bg.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            }
            if quit_bg.load(Ordering::Relaxed) {
                break;
            }
        }
    });

    #[allow(unused_assignments)]
    let mut status = String::new();
    let bench_start = Instant::now();
    let total_steps = target_sizes.len() * 3;
    let mut completed_steps = 0usize;

    let mut order_rng = StdRng::seed_from_u64(0x105E27);

    'outer: for &size in &target_sizes {
        let mut order = [0u8, 1, 2];
        order.shuffle(&mut order_rng);

        for &idx in &order {
            if quit.load(Ordering::Relaxed) {
                break 'outer;
            }

            let impl_name = match idx {
                0 => "pomap",
                1 => "std_hashmap",
                _ => "hashbrown",
            };

            let eta = if completed_steps > 0 {
                let elapsed = bench_start.elapsed().as_secs_f64();
                let per_step = elapsed / completed_steps as f64;
                let remaining = (total_steps - completed_steps) as f64 * per_step;
                if remaining >= 3600.0 {
                    format!("{:.1}h", remaining / 3600.0)
                } else if remaining >= 60.0 {
                    format!("{:.0}m", remaining / 60.0)
                } else {
                    format!("{:.0}s", remaining)
                }
            } else {
                "...".to_string()
            };
            status = format!(
                " size={:<6} | {} | {}/{} | ETA {} | 'q' or Ctrl+C to stop",
                size,
                impl_name,
                completed_steps + 1,
                total_steps,
                eta
            );
            draw_tui(&mut terminal, &chart_data, &status);

            match idx {
                0 => time_inserts!(
                    size,
                    "pomap",
                    BenchPoMap::with_hasher(BenchHasherBuilder::default())
                ),
                1 => time_inserts!(
                    size,
                    "std_hashmap",
                    BenchHashMap::with_hasher(BenchHasherBuilder::default())
                ),
                _ => time_inserts!(
                    size,
                    "hashbrown",
                    BenchHashbrownMap::with_hasher(BenchHasherBuilder::default())
                ),
            }

            completed_steps += 1;
        }
    }

    if !quit.load(Ordering::Relaxed) {
        draw_tui(&mut terminal, &chart_data, " Done! Press any key to exit.");
        while !quit.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    disable_raw_mode().unwrap();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).unwrap();
}
