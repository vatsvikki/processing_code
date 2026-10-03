import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Minimal test: chain function blocks into a pipe

    Set a **starting value**, then click **➕ Addition** or
    **✖️ Multiplication** in the top bar to append that step to the
    pipe -- each click adds one more step, using whatever is currently
    in **next value (b)**. Click a block again and again to add
    several steps in a row. **▶ Run pipe** then runs the whole chain in
    order, each step feeding the next, and shows every intermediate
    result on the right.
    """)
    return


@app.cell
def _(mo):
    get_pipe, set_pipe = mo.state([])  # list of {"op": "add"|"multiply", "b": number}
    get_trace, set_trace = mo.state(None)  # list of (label, running_value) or None
    return get_pipe, get_trace, set_pipe, set_trace


@app.cell
def _(mo):
    add_block_button = mo.ui.run_button(label="➕ Addition")
    mul_block_button = mo.ui.run_button(label="✖️ Multiplication")
    clear_button = mo.ui.run_button(label="🗑️ Clear pipe", kind="danger")
    return add_block_button, clear_button, mul_block_button


@app.cell
def _(mo):
    input_start = mo.ui.number(value=0, label="Starting value (a)")
    input_b = mo.ui.number(value=1, label="Next value (b) -- used when you click a block")
    run_button = mo.ui.run_button(label="▶ Run pipe")
    return input_b, input_start, run_button


@app.cell
def _(
    add_block_button,
    clear_button,
    get_pipe,
    input_b,
    mul_block_button,
    set_pipe,
    set_trace,
):
    # Appending/clearing changes the pipe -- any previous run's trace is
    # now stale, so clear it too (avoids showing an old result next to
    # a pipe that no longer matches it).
    if add_block_button.value:
        set_pipe(list(get_pipe()) + [{"op": "add", "b": input_b.value}])
        set_trace(None)
    elif mul_block_button.value:
        set_pipe(list(get_pipe()) + [{"op": "multiply", "b": input_b.value}])
        set_trace(None)
    elif clear_button.value:
        set_pipe([])
        set_trace(None)
    return


@app.cell
def _(get_pipe, input_start, run_button, set_trace):
    if run_button.value:
        _pipe = get_pipe()
        _val = input_start.value
        _trace = [("start", _val)]
        for _step in _pipe:
            if _step["op"] == "add":
                _val = _val + _step["b"]
                _label = f"+ {_step['b']}"
            else:
                _val = _val * _step["b"]
                _label = f"× {_step['b']}"
            _trace.append((_label, _val))
        set_trace(_trace)
    return


@app.cell
def _(add_block_button, clear_button, mo, mul_block_button):
    # Top bar -- 10% of screen height, full width. Function blocks live
    # here; each click appends one more step to the pipe below.
    mo.Html(f"""
    <div style="
        height: 10vh; min-height: 56px; width: 100%; box-sizing: border-box;
        display: flex; align-items: center; gap: 12px; padding: 0 16px;
        border-bottom: 1px solid #ccc;
    ">
      {add_block_button}
      {mul_block_button}
      {clear_button}
    </div>
    """)
    return


@app.cell
def _(get_pipe, input_b, input_start, mo, run_button):
    # Left column -- 15% of screen width: build the pipe here (starting
    # value, next-value field, the run/clear controls already live in
    # the top bar) and see the current pipe as a plain numbered list.
    _pipe = get_pipe()
    if _pipe:
        _steps_md = "\n".join(
            f"{_i + 1}. {'+' if _s['op'] == 'add' else '×'} {_s['b']}"
            for _i, _s in enumerate(_pipe)
        )
        _pipe_display = mo.md(f"**Pipe ({len(_pipe)} step(s)):**\n\n{_steps_md}")
    else:
        _pipe_display = mo.md("*Pipe is empty -- click a block above to add a step.*")

    left_panel_content = mo.vstack([
        input_start,
        input_b,
        mo.md("---"),
        _pipe_display,
        run_button,
    ])
    return (left_panel_content,)


@app.cell
def _(get_trace, left_panel_content, mo):
    # Right area -- everything else: the step-by-step trace and final
    # result once the pipe has been run.
    _trace = get_trace()
    if _trace is None:
        _right_content = mo.md("*Build a pipe on the left, then click **Run pipe**.*")
    else:
        _trace_md = "\n".join(
            f"- {_label}: **{_val}**" for _label, _val in _trace
        )
        _right_content = mo.vstack([
            mo.md(f"### Trace\n{_trace_md}"),
            mo.callout(f"Final result = **{_trace[-1][1]}**", kind="success"),
        ])

    mo.Html(f"""
    <div style="display: flex; flex-direction: row; width: 100%; align-items: flex-start;">
      <div style="width: 15%; min-width: 160px; box-sizing: border-box; padding: 16px; border-right: 1px solid #ccc;">
        {left_panel_content}
      </div>
      <div style="flex: 1; box-sizing: border-box; padding: 16px;">
        {_right_content}
      </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
