r"""
easy_act_csv.py

基于 pandas 的 CSV 数据处理小工具（高效：支持 chunksize 分块处理）。

功能：
- 新增列：常量 / 按列拼接 / 按模板格式化
- 合并列：按分隔符或模板组合成新列（可选删除源列）
- 抽取列：从原 CSV 选择特定列写到新 CSV（可自定义输出列名）
- 原位替换列内容：对已有列整列覆盖（常量 / 按列拼接 / 按模板格式化）
- 删除列：删除指定的列
- 路径替换：对指定列（如 image_path）做子串/正则替换（常用于修复挂载前缀）
- pipeline/multi：一次命令顺序执行多个子操作（适合对同一 CSV 连续加工）

关键约束（按你的要求）：
- **且仅对新增的字符串内容强制包裹双引号 ""**，避免后续读取误解析
- 其它列只做 CSV 的“最小必要引号”（遇到分隔符/引号/换行才加），不做全表包裹

If exist:
使用 --if-exists 参数来控制当新列已存在时的处理方式：
- replace: 覆盖
- error: 报错
- skip: 跳过


示例：
  新增常量列（只对 new_col 的字符串强制加引号）：
    python Code/Transf/easy_act_csv.py -i in.csv -o out.csv addcol --new-col new_col --value "一句话"

  原位替换列内容（将已存在的 instruction 整列覆盖为常量；默认按“最小必要引号”写出）：
    python Code/Transf/easy_act_csv.py -i in.csv -o out.csv setcol --col instruction --value "一句话"

  合并列（a,b 用下划线拼接成 ab；可删除 a,b）：
    python Code/Transf/easy_act_csv.py -i in.csv -o out.csv mergecols --new-col ab --cols a b --sep "_" --drop

  原地覆盖写回：
    python Code/Transf/easy_act_csv.py -i in.csv --inplace addcol --new-col new_col --value "一句话"

  顺序 index 列 默认插在第一列（支持起始 seed 如 "00000"）
  python Code/Transf/easy_act_csv.py -i in.csv -o out.csv addindex --new-col index --seed "00000"
  python Code/Transf/easy_act_csv.py -i in.csv --inplace addindex --new-col index --seed "00000"

  删除列：
    python Code/Transf/easy_act_csv.py -i in.csv -o out.csv dropcol --col col_name
    python3 easy_act_csv.py -i in.csv -o out.csv dropcol --col col_name

  抽取列到新 CSV（可重命名列名）：
    python3 easy_act_csv.py -i in.csv -o out.csv extractcols --cols image_path instruction
    python3 easy_act_csv.py -i in.csv -o out.csv extractcols --cols image_path instruction --map instruction:prompt
    python3 easy_act_csv.py -i in.csv -o out.csv extractcols --map image_path:img --map instruction:prompt

  路径前缀替换（常用：把旧前缀替换成新前缀）：
    python3 easy_act_csv.py -i in.csv -o out.csv replacepath --col image_path \
      --old "OLD_PREFIX/" --new "NEW_PREFIX/"

  正则替换（例如把 Windows 反斜杠统一成 /）：
    python3 easy_act_csv.py -i in.csv -o out.csv replacepath --col image_path \
      --regex --old "\\\\" --new "/"

  统一两种 editplan JSON 格式为标准字段，并写入新列：
    python3 easy_act_csv.py -i in.csv -o out.csv unifyeditplan --src-col editplan --new-col editplan_unified
    python3 easy_act_csv.py -i in.csv -o out.csv unifyeditplan --src-col editplan --new-col editplan_unified --if-src-missing error
    python3 easy_act_csv.py -i in.csv -o out.csv unifyeditplan --src-col editplan --new-col editplan_unified --if-src-missing skip
    python3 easy_act_csv.py -i in.csv -o out.csv unifyeditplan --src-col editplan --new-col editplan_unified --if-src-missing skip --if-exists replace
    python3 easy_act_csv.py -i in.csv -o out.csv unifyeditplan --src-col editplan --new-col editplan_unified --if-src-missing skip --if-exists skip
    python3 easy_act_csv.py -i in.csv -o out.csv unifyeditplan --src-col editplan --new-col editplan_unified --if-src-missing skip --if-exists skip

  提取并写入到新 CSV（可重命名列名）：
    python3 easy_act_csv.py -i in.csv -o out.csv extractcols --cols image_path instruction
    python3 easy_act_csv.py -i in.csv -o out.csv extractcols --cols image_path instruction --map instruction:prompt
    python3 easy_act_csv.py -i in.csv -o out.csv extractcols --map image_path:img --map instruction:prompt


  pipeline/multi（一次命令串多个操作；每个步骤以子命令名开头）：
    python3 easy_act_csv.py \
      -i data/Line.csv \
      --inplace \
      --if-exists replace \
      pipeline \
        addcol --new-col preinst --value "Edit the image to show the impact of this factor:" \
        mergecols --new-col driving_factor --cols driving_factor_text --sep "" \
        mergecols --new-col instruction --cols preinst driving_factor --sep " " --drop \
        replacepath --col image_path --old "OLD_PREFIX/" --new "NEW_PREFIX/" \
        dropcol --col factor_type --if-exists skip


（更多复杂示例省略；建议从 repo 根目录相对路径调用，例如：`python Preprocess/easy_act_csv.py ...`）


"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import tempfile
import shutil
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd


_NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _looks_like_int_with_leading_zeros(s: str) -> bool:
    s2 = s.strip()
    if not s2:
        return False
    if s2[0] in "+-":
        s2 = s2[1:]
    if len(s2) <= 1:
        return False
    return s2[0] == "0" and s2[1].isdigit()


def _is_numeric_string(s: str) -> bool:
    s2 = s.strip()
    if s2 == "":
        return False
    if _looks_like_int_with_leading_zeros(s2):
        return False
    if not _NUM_RE.match(s2):
        return False
    return True


def _format_csv_field(
    s: str,
    *,
    delimiter: str,
    quotechar: str,
    always_quote: bool,
    minimal_quote: bool,
) -> str:
    # RFC4180 风格：quotechar 内部用双写转义
    if always_quote:
        return quotechar + s.replace(quotechar, quotechar * 2) + quotechar
    if minimal_quote and any(ch in s for ch in (delimiter, quotechar, "\n", "\r")):
        return quotechar + s.replace(quotechar, quotechar * 2) + quotechar
    return s


def _write_csv_row(
    out_f,
    row: Sequence[str],
    *,
    delimiter: str,
    quotechar: str,
    lineterminator: str,
    new_col_names: Sequence[str],
    new_col_value_type: str,  # auto|str|int|float
) -> None:
    """
    逐字段写出：
    - 对新增列：如果是“字符串内容”，强制加引号
    - 对其它列：最小必要引号
    """
    new_set = set(new_col_names)
    parts: List[str] = []
    for col_name, cell in zip(row[0::2], row[1::2]):
        # row 这里采用 (colname, value, colname, value, ...) 形式，方便判断是否新增列
        v = "" if cell is None else str(cell)

        if col_name in new_set:
            if v == "":
                # 空值不强制引号（""/空字段语义等价；这里尽量少改动）
                parts.append(_format_csv_field(v, delimiter=delimiter, quotechar=quotechar, always_quote=False, minimal_quote=True))
                continue

            if new_col_value_type == "str":
                should_quote = True
            elif new_col_value_type in {"int", "float"}:
                should_quote = False
            else:
                # auto：像数字则不加引号，否则加
                should_quote = not _is_numeric_string(v)

            parts.append(
                _format_csv_field(
                    v,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    always_quote=should_quote,
                    minimal_quote=False,
                )
            )
        else:
            parts.append(
                _format_csv_field(
                    v,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    always_quote=False,
                    minimal_quote=True,
                )
            )

    out_f.write(delimiter.join(parts) + lineterminator)


def _iter_chunks(
    input_path: str,
    *,
    encoding: str,
    delimiter: str,
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    """
    全部读成字符串，尽量保持原样；避免 NA 被转成 NaN。

    兼容性：
    - 默认先用 pandas 的 C engine（更快）
    - 若遇到 CSV 行字段数不一致等导致的 ParserError，则自动降级到 python engine，并用 on_bad_lines='warn'
      （坏行会被跳过，并输出 warning；适用于部分“脏 CSV”）
    """

    common_kwargs = dict(
        encoding=encoding,
        sep=delimiter,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=chunksize,
    )

    def _make_reader(engine: str, on_bad_lines: str):
        try:
            return pd.read_csv(input_path, engine=engine, on_bad_lines=on_bad_lines, **common_kwargs)
        except TypeError:
            # 兼容老版本 pandas：可能不支持 on_bad_lines 参数
            return pd.read_csv(input_path, engine=engine, **common_kwargs)

    # 先尝试 C engine
    reader = _make_reader("c", "error")
    try:
        first = next(reader)  # type: ignore[arg-type]
        yield first
        for chunk in reader:
            yield chunk
        return
    except StopIteration:
        return
    except Exception as e:
        # CSV 内容不规整时 C engine 会抛 ParserError；改用 python engine 宽容读取
        if isinstance(e, pd.errors.ParserError):
            import sys

            print(
                f"[WARN] pandas ParserError while reading CSV; fallback to engine='python' on_bad_lines='warn': {input_path}",
                file=sys.stderr,
            )
            reader2 = _make_reader("python", "warn")
            for chunk in reader2:
                yield chunk
            return
        raise


def _parse_rename_maps(maps: Optional[Sequence[str]]) -> Dict[str, str]:
    """
    解析列重命名映射，支持两种写法：
      - old:new
      - old=new
    可重复传入多次；后面的覆盖前面的。
    """
    if not maps:
        return {}
    out: Dict[str, str] = {}
    for raw in maps:
        s = str(raw).strip()
        if not s:
            continue
        if ":" in s:
            old, new = s.split(":", 1)
        elif "=" in s:
            old, new = s.split("=", 1)
        else:
            raise ValueError(f"--map 语法错误，必须是 old:new 或 old=new；收到: {raw!r}")
        old = old.strip()
        new = new.strip()
        if not old or not new:
            raise ValueError(f"--map 语法错误，old/new 不能为空；收到: {raw!r}")
        out[old] = new
    return out


def extract_columns_to_new_csv(
    *,
    input_path: str,
    output_path: str,
    cols: Optional[List[str]] = None,
    rename_maps: Optional[Sequence[str]] = None,
    if_missing: str = "error",  # error|skip
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
    chunksize: int = 200_000,
) -> None:
    """
    从原 CSV 中抽取指定列写入到新的 CSV 文件中，并支持自定义输出列名。

    - cols: 要抽取的源列名（按给定顺序写出）
    - rename_maps: --map old:new / old=new 的列表，用于重命名输出表头（值不变）
    - if_missing:
        - error: 任一指定列不存在则报错
        - skip: 忽略不存在的列（至少保留 1 列，否则报错）
    """
    if if_missing not in {"error", "skip"}:
        raise ValueError("if_missing 只能是: error / skip")

    rename = _parse_rename_maps(rename_maps)
    if cols is None:
        # 若没显式给 --cols，则从 --map 的 key 推断要抽取的列（按给定顺序）
        if rename:
            cols = list(rename.keys())
        else:
            raise ValueError("extractcols 必须指定 --cols 或至少一个 --map")

    # 去掉空列名（容错）
    cols = [c for c in (cols or []) if str(c).strip() != ""]
    if not cols:
        raise ValueError("extractcols: 需要至少一个列名")

    # 快速检查列存在性（并按 if_missing 处理）
    try:
        cols0 = pd.read_csv(
            input_path,
            encoding=encoding,
            sep=delimiter,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            nrows=0,
        ).columns
        missing = [c for c in cols if c not in cols0]
        if missing:
            if if_missing == "skip":
                cols = [c for c in cols if c in cols0]
            else:
                raise ValueError(f"缺少列：{missing}")
    except Exception as e:
        if "缺少列：" in str(e):
            raise
        # 解析失败就走常规流程，让后续读/写给出更明确的错误
        pass

    if not cols:
        raise ValueError("extractcols: 在 if_missing=skip 下没有任何可抽取的列")

    out_cols = [rename.get(c, c) for c in cols]
    # 防止输出表头重复
    if len(set(out_cols)) != len(out_cols):
        raise ValueError(f"extractcols: 输出列名重复（请检查 --map）：{out_cols}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    quotechar = '"'
    lineterminator = "\n"
    wrote_header = False

    with open(output_path, "w", encoding=encoding, newline="") as f_out:
        for chunk in _iter_chunks(input_path, encoding=encoding, delimiter=delimiter, chunksize=chunksize):
            # 注意：这里选择列/重命名只影响写出结果，不改写输入数据本身
            # 若在分块中出现缺失（例如脏文件），按 if_missing 处理
            exist_cols = [c for c in cols if c in chunk.columns]
            missing2 = [c for c in cols if c not in chunk.columns]
            if missing2 and if_missing == "error":
                raise ValueError(f"缺少列：{missing2}")
            if not exist_cols:
                # 本块没有任何可抽取列：跳过
                continue

            out_chunk = chunk[exist_cols].copy()
            out_chunk.rename(columns=rename, inplace=True)

            # 写表头
            if not wrote_header:
                header_cells = list(out_chunk.columns)
                header_line = delimiter.join(
                    _format_csv_field(str(h), delimiter=delimiter, quotechar=quotechar, always_quote=False, minimal_quote=True)
                    for h in header_cells
                )
                f_out.write(header_line + lineterminator)
                wrote_header = True

            # 写数据行：抽取/重命名不属于新增内容 -> 所有列最小必要引号
            col_names = list(out_chunk.columns)
            for tup in out_chunk.itertuples(index=False, name=None):
                interleaved: List[str] = []
                for c, v in zip(col_names, tup):
                    interleaved.append(c)
                    interleaved.append("" if v is None else str(v))
                _write_csv_row(
                    f_out,
                    interleaved,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    lineterminator=lineterminator,
                    new_col_names=[],
                    new_col_value_type="auto",
                )

    if not wrote_header:
        # 极端情况：输入为空或全被 skip 掉，仍写出一个空文件不太友好
        raise ValueError("extractcols: 未写出任何数据（可能是输入为空，或所有列都缺失/被跳过）")

def _compute_new_col(
    chunk: pd.DataFrame,
    *,
    value: Optional[str],
    from_cols: Optional[List[str]],
    sep: str,
    template: Optional[str],
) -> pd.Series:
    if value is not None:
        return pd.Series([value] * len(chunk), index=chunk.index)

    if template is not None:
        # 注意：模板格式化需要逐行处理，性能较拼接差一些
        def fmt_row(row) -> str:
            d = row.to_dict()
            return template.format(**d)

        return chunk.apply(fmt_row, axis=1)

    if from_cols:
        missing = [c for c in from_cols if c not in chunk.columns]
        if missing:
            raise ValueError(f"缺少列：{missing}")
        # 按行拼接
        return chunk[from_cols].astype(str).agg(sep.join, axis=1)

    raise ValueError("必须指定 --value 或 --cols/--sep 或 --template 之一")


def _parse_jsonish_cell(s: Any) -> Any:
    """
    解析 CSV 单元格中的“类 JSON”字符串：
    - 优先 json.loads
    - 失败则尝试 ast.literal_eval（兼容单引号/None 等）
    - 空/缺失返回 None
    """
    if s is None:
        return None
    if isinstance(s, (list, dict)):
        return s
    text = str(s).strip()
    if text == "" or text.lower() in {"null", "none", "nan"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return None


def _none_if_empty_list(v: Any) -> Any:
    if isinstance(v, list) and len(v) == 0:
        return None
    return v


def _unify_editplan_payload(payload: Any) -> Any:
    """
    将两种 editplan 格式统一到：
      [{"object_id": 1, "object_name": ..., "status": ..., "intention": ..., "actions": ..., "rules": ...}, ...]
    缺失字段填 None（序列化为 JSON 的 null）。
    """
    if not isinstance(payload, list):
        return None

    unified: List[Dict[str, Any]] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            # 非 dict 项直接跳过（避免污染输出）；需要的话可改成全 null 记录
            continue

        # format1: object_name / intention / actions / rules
        # format2: object / change_spec.intention / change_spec.actions
        change_spec = item.get("change_spec")
        if not isinstance(change_spec, dict):
            change_spec = {}

        object_name = item.get("object_name", item.get("object"))
        status = item.get("status")

        intention = item.get("intention", None)
        if intention is None:
            intention = change_spec.get("intention", None)

        actions = item.get("actions", None)
        if actions is None:
            actions = change_spec.get("actions", None)

        rules = item.get("rules", None)
        if rules is None:
            rules = change_spec.get("rules", None)

        unified.append(
            {
                "object_id": idx,
                "object_name": object_name if object_name is not None else None,
                "status": status if status is not None else None,
                "intention": intention if intention is not None else None,
                "actions": _none_if_empty_list(actions),
                "rules": _none_if_empty_list(rules),
            }
        )

    return unified


def unify_editplan_column_in_csv(
    *,
    input_path: str,
    output_path: str,
    src_col: str,
    new_col: str,
    if_src_missing: str = "error",  # error|skip
    if_exists: str = "error",  # error|replace|skip
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
    chunksize: int = 200_000,
) -> None:
    """
    从 src_col 读取两种格式的 JSON 列表字符串，统一为固定 keys 的 JSON 列表，写入 new_col。
    - object_id 按列表顺序从 1 开始重新编号
    - 缺失字段填 null
    """
    if if_src_missing not in {"error", "skip"}:
        raise ValueError("if_src_missing 只能是: error / skip")
    if if_exists not in {"error", "replace", "skip"}:
        raise ValueError("if_exists 只能是: error / replace / skip")

    # 快速检查源列存在性
    try:
        cols0 = pd.read_csv(
            input_path,
            encoding=encoding,
            sep=delimiter,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            nrows=0,
        ).columns
        if src_col not in cols0:
            if if_src_missing == "skip":
                os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
                shutil.copyfile(input_path, output_path)
                return
            raise ValueError(f"源列不存在: {src_col}")
    except Exception as e:
        if "源列不存在" in str(e):
            raise
        # 解析失败就走常规流程，让后续读/写给出更明确的错误
        pass

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    quotechar = '"'
    lineterminator = "\n"
    wrote_header = False

    with open(output_path, "w", encoding=encoding, newline="") as f_out:
        for chunk in _iter_chunks(input_path, encoding=encoding, delimiter=delimiter, chunksize=chunksize):
            chunk = chunk.copy()

            if src_col not in chunk.columns:
                if if_src_missing == "skip":
                    # 不改动（但会标准化写出格式）
                    pass
                else:
                    raise ValueError(f"源列不存在: {src_col}")
            else:
                if new_col in chunk.columns:
                    if if_exists == "skip":
                        pass
                    elif if_exists == "replace":
                        pass
                    else:
                        raise ValueError(f"列已存在: {new_col}（可用 --if-exists replace/skip）")

                def _conv(cell: Any) -> str:
                    payload = _parse_jsonish_cell(cell)
                    unified = _unify_editplan_payload(payload)
                    if unified is None:
                        return "null"
                    return json.dumps(unified, ensure_ascii=False, separators=(",", ":"))

                chunk[new_col] = chunk[src_col].apply(_conv)

            # 写表头
            if not wrote_header:
                header_cells = list(chunk.columns)
                header_line = delimiter.join(
                    _format_csv_field(str(h), delimiter=delimiter, quotechar=quotechar, always_quote=False, minimal_quote=True)
                    for h in header_cells
                )
                f_out.write(header_line + lineterminator)
                wrote_header = True

            # 写数据行：new_col 属于新增/替换字符串列 -> 强制加引号
            col_names = list(chunk.columns)
            for tup in chunk.itertuples(index=False, name=None):
                interleaved: List[str] = []
                for c, v in zip(col_names, tup):
                    interleaved.append(c)
                    interleaved.append("" if v is None else str(v))
                _write_csv_row(
                    f_out,
                    interleaved,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    lineterminator=lineterminator,
                    new_col_names=[new_col],
                    new_col_value_type="str",
                )


def _extract_object_list_from_unified_editplan(payload: Any, *, dedupe: bool) -> List[str]:
    """
    从“标准字段 editplan”（list[dict]）中抽取 object list。
    约定：优先使用 item["object_name"]，否则回退到 item["object"]。

    返回：按原顺序的对象名列表（可选去重）。
    """
    if not isinstance(payload, list):
        return []

    out: List[str] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = item.get("object_name", None)
        if name is None:
            name = item.get("object", None)
        if name is None:
            continue
        s = str(name).strip()
        if s == "":
            continue
        if dedupe:
            if s in seen:
                continue
            seen.add(s)
        out.append(s)
    return out


def extract_object_list_from_editplan_in_csv(
    *,
    input_path: str,
    output_path: str,
    src_col: str,
    new_col: str,
    dedupe: bool = True,
    if_src_missing: str = "error",  # error|skip
    if_exists: str = "error",  # error|replace|skip
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
    chunksize: int = 200_000,
) -> None:
    """
    从标准字段 editplan JSON 列中提取 object list，并写入新列（JSON 数组字符串）。

    输入列内容期望为：
      [{"object_id":1,"object_name":"A",...}, ...]

    输出列例如：
      ["A","B","C"]
    """
    if if_src_missing not in {"error", "skip"}:
        raise ValueError("if_src_missing 只能是: error / skip")
    if if_exists not in {"error", "replace", "skip"}:
        raise ValueError("if_exists 只能是: error / replace / skip")

    # 快速检查源列存在性
    try:
        cols0 = pd.read_csv(
            input_path,
            encoding=encoding,
            sep=delimiter,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            nrows=0,
        ).columns
        if src_col not in cols0:
            if if_src_missing == "skip":
                os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
                shutil.copyfile(input_path, output_path)
                return
            raise ValueError(f"源列不存在: {src_col}")
    except Exception as e:
        if "源列不存在" in str(e):
            raise
        pass

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    quotechar = '"'
    lineterminator = "\n"
    wrote_header = False

    with open(output_path, "w", encoding=encoding, newline="") as f_out:
        for chunk in _iter_chunks(input_path, encoding=encoding, delimiter=delimiter, chunksize=chunksize):
            chunk = chunk.copy()

            if src_col not in chunk.columns:
                if if_src_missing == "skip":
                    pass
                else:
                    raise ValueError(f"源列不存在: {src_col}")
            else:
                if new_col in chunk.columns:
                    if if_exists == "skip":
                        pass
                    elif if_exists == "replace":
                        pass
                    else:
                        raise ValueError(f"列已存在: {new_col}（可用 --if-exists replace/skip）")

                def _conv(cell: Any) -> str:
                    payload = _parse_jsonish_cell(cell)
                    objs = _extract_object_list_from_unified_editplan(payload, dedupe=dedupe)
                    return json.dumps(objs, ensure_ascii=False, separators=(",", ":"))

                chunk[new_col] = chunk[src_col].apply(_conv)

            # 写表头
            if not wrote_header:
                header_cells = list(chunk.columns)
                header_line = delimiter.join(
                    _format_csv_field(str(h), delimiter=delimiter, quotechar=quotechar, always_quote=False, minimal_quote=True)
                    for h in header_cells
                )
                f_out.write(header_line + lineterminator)
                wrote_header = True

            # 写数据行：new_col 属于新增/替换字符串列 -> 强制加引号
            col_names = list(chunk.columns)
            for tup in chunk.itertuples(index=False, name=None):
                interleaved: List[str] = []
                for c, v in zip(col_names, tup):
                    interleaved.append(c)
                    interleaved.append("" if v is None else str(v))
                _write_csv_row(
                    f_out,
                    interleaved,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    lineterminator=lineterminator,
                    new_col_names=[new_col],
                    new_col_value_type="str",
                )


def _parse_index_seed(seed: str) -> tuple[int, int]:
    """
    解析形如 "00000" / "00012" 的 seed：
    - 宽度 = len(seed)
    - 起始数值 = int(seed)
    """
    s = str(seed).strip()
    if not re.fullmatch(r"\d+", s):
        raise ValueError(f"--seed 必须是纯数字字符串，例如 00000；实际为: {seed!r}")
    return int(s), len(s)


def add_index_to_csv(
    *,
    input_path: str,
    output_path: str,
    index_col: str = "index",
    seed: str = "00000",
    step: int = 1,
    pos: str = "first",  # first|last
    if_exists: str = "error",  # error|replace|skip
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
    chunksize: int = 200_000,
) -> None:
    if pos not in {"first", "last"}:
        raise ValueError("pos 只能是: first / last")
    if if_exists not in {"error", "replace", "skip"}:
        raise ValueError("if_exists 只能是: error / replace / skip")
    if step == 0:
        raise ValueError("--step 不能为 0")

    # 若选择 skip 且列已存在：直接复制文件，避免改动其它列的引号/格式
    try:
        cols0 = pd.read_csv(
            input_path,
            encoding=encoding,
            sep=delimiter,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            nrows=0,
        ).columns
        if index_col in cols0 and if_exists == "skip":
            os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
            shutil.copyfile(input_path, output_path)
            return
    except Exception:
        # 解析失败就走常规流程，让后续读/写给出更明确的错误
        pass

    start, width = _parse_index_seed(seed)
    cur = start

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    quotechar = '"'
    lineterminator = "\n"
    wrote_header = False

    with open(output_path, "w", encoding=encoding, newline="") as f_out:
        for chunk in _iter_chunks(input_path, encoding=encoding, delimiter=delimiter, chunksize=chunksize):
            chunk = chunk.copy()

            if index_col in chunk.columns:
                if if_exists == "skip":
                    # 上面已 copyfile 并 return；这里兜底
                    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
                    shutil.copyfile(input_path, output_path)
                    return
                if if_exists == "replace":
                    pass
                else:
                    raise ValueError(f"列已存在: {index_col}（可用 --if-exists replace/skip）")

            n = len(chunk)
            # 生成顺序索引（字符串），例如 00000,00001,...
            idx_values = [f"{cur + step * i:0{width}d}" for i in range(n)]
            cur = cur + step * n

            if index_col in chunk.columns and if_exists == "replace":
                chunk[index_col] = idx_values
            else:
                if pos == "first":
                    chunk.insert(0, index_col, idx_values)
                else:
                    chunk[index_col] = idx_values

            # 写表头
            if not wrote_header:
                header_cells = list(chunk.columns)
                header_line = delimiter.join(
                    _format_csv_field(
                        str(h),
                        delimiter=delimiter,
                        quotechar=quotechar,
                        always_quote=False,
                        minimal_quote=True,
                    )
                    for h in header_cells
                )
                f_out.write(header_line + lineterminator)
                wrote_header = True

            # 写数据行：index_col 是新增字符串列 -> 强制加引号
            col_names = list(chunk.columns)
            for tup in chunk.itertuples(index=False, name=None):
                interleaved: List[str] = []
                for c, v in zip(col_names, tup):
                    interleaved.append(c)
                    interleaved.append("" if v is None else str(v))
                _write_csv_row(
                    f_out,
                    interleaved,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    lineterminator=lineterminator,
                    new_col_names=[index_col],
                    new_col_value_type="str",
                )


def process_csv(
    *,
    input_path: str,
    output_path: str,
    op: str,  # addcol|mergecols
    new_col: str,
    value: Optional[str],
    cols: Optional[List[str]],
    sep: str,
    template: Optional[str],
    drop: bool,
    if_exists: str,  # error|replace|skip
    encoding: str,
    delimiter: str,
    chunksize: int,
    new_col_value_type: str,  # auto|str|int|float
) -> None:
    if if_exists not in {"error", "replace", "skip"}:
        raise ValueError("if_exists 只能是: error / replace / skip")
    if new_col_value_type not in {"auto", "str", "int", "float"}:
        raise ValueError("new_col_value_type 只能是: auto / str / int / float")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    quotechar = '"'
    lineterminator = "\n"
    wrote_header = False

    with open(output_path, "w", encoding=encoding, newline="") as f_out:
        for chunk in _iter_chunks(input_path, encoding=encoding, delimiter=delimiter, chunksize=chunksize):
            chunk = chunk.copy()

            if new_col in chunk.columns:
                if if_exists == "skip":
                    # 不做任何修改：直接把原 chunk 写出（但写出格式会标准化）
                    pass
                elif if_exists == "replace":
                    chunk[new_col] = _compute_new_col(
                        chunk, value=value, from_cols=cols, sep=sep, template=template
                    )
                else:
                    raise ValueError(f"列已存在: {new_col}（可用 --if-exists replace/skip）")
            else:
                chunk[new_col] = _compute_new_col(
                    chunk, value=value, from_cols=cols, sep=sep, template=template
                )

            if op == "mergecols" and drop and cols:
                # 删除源列（但别误删新列同名情况）
                drop_cols = [c for c in cols if c != new_col]
                chunk.drop(columns=[c for c in drop_cols if c in chunk.columns], inplace=True)

            # 写表头
            if not wrote_header:
                header_cells = list(chunk.columns)
                # 表头不属于“新增内容”，统一最小必要引号
                header_line = delimiter.join(
                    _format_csv_field(str(h), delimiter=delimiter, quotechar=quotechar, always_quote=False, minimal_quote=True)
                    for h in header_cells
                )
                f_out.write(header_line + lineterminator)
                wrote_header = True

            # 写数据行：为了做到“仅新增列强制引号”，这里用自定义逐字段写
            col_names = list(chunk.columns)
            new_cols = [new_col]  # 当前工具一次只新增/替换一个目标列

            # itertuples 比 iterrows 快
            for tup in chunk.itertuples(index=False, name=None):
                # 变成 (colname, value, ...) 结构供 _write_csv_row 判断
                interleaved: List[str] = []
                for c, v in zip(col_names, tup):
                    interleaved.append(c)
                    interleaved.append("" if v is None else str(v))
                _write_csv_row(
                    f_out,
                    interleaved,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    lineterminator=lineterminator,
                    new_col_names=new_cols,
                    new_col_value_type=new_col_value_type,
                )


def drop_column_from_csv(
    *,
    input_path: str,
    output_path: str,
    col_name: str,
    if_exists: str = "error",  # error|skip
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
    chunksize: int = 200_000,
) -> None:
    """
    从 CSV 文件中删除指定列（分块处理）。
    """
    if if_exists not in {"error", "skip"}:
        raise ValueError("if_exists 只能是: error / skip")

    # 先检查列是否存在
    try:
        cols0 = pd.read_csv(
            input_path,
            encoding=encoding,
            sep=delimiter,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            nrows=0,
        ).columns
        if col_name not in cols0:
            if if_exists == "skip":
                # 列不存在且选择 skip：直接复制文件
                os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
                shutil.copyfile(input_path, output_path)
                return
            else:
                raise ValueError(f"列不存在: {col_name}")
    except Exception as e:
        if "列不存在" in str(e):
            raise
        # 解析失败就走常规流程，让后续读/写给出更明确的错误
        pass

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    quotechar = '"'
    lineterminator = "\n"
    wrote_header = False

    with open(output_path, "w", encoding=encoding, newline="") as f_out:
        for chunk in _iter_chunks(input_path, encoding=encoding, delimiter=delimiter, chunksize=chunksize):
            chunk = chunk.copy()

            if col_name not in chunk.columns:
                if if_exists == "skip":
                    # 列不存在：直接写出当前 chunk（但写出格式会标准化）
                    pass
                else:
                    raise ValueError(f"列不存在: {col_name}")
            else:
                # 删除列
                chunk.drop(columns=[col_name], inplace=True)

            # 写表头
            if not wrote_header:
                header_cells = list(chunk.columns)
                header_line = delimiter.join(
                    _format_csv_field(
                        str(h),
                        delimiter=delimiter,
                        quotechar=quotechar,
                        always_quote=False,
                        minimal_quote=True,
                    )
                    for h in header_cells
                )
                f_out.write(header_line + lineterminator)
                wrote_header = True

            # 写数据行：删除列不涉及新增内容，所以所有列都用最小必要引号
            col_names = list(chunk.columns)
            for tup in chunk.itertuples(index=False, name=None):
                interleaved: List[str] = []
                for c, v in zip(col_names, tup):
                    interleaved.append(c)
                    interleaved.append("" if v is None else str(v))
                _write_csv_row(
                    f_out,
                    interleaved,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    lineterminator=lineterminator,
                    new_col_names=[],  # 删除列不涉及新增，所以为空
                    new_col_value_type="auto",
                )


_PIPE_OPS = (
    "addcol",
    "mergecols",
    "setcol",
    "addindex",
    "dropcol",
    "replacepath",
    "unifyeditplan",
    "extractcols",
    "extractobjlist",
)


def set_column_in_csv(
    *,
    input_path: str,
    output_path: str,
    col_name: str,
    value: Optional[str],
    cols: Optional[List[str]],
    sep: str,
    template: Optional[str],
    if_missing: str = "error",  # error|skip|create
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
    chunksize: int = 200_000,
    col_value_type: str = "auto",  # auto|str|int|float
    quote_mode: str = "minimal",  # minimal|newcol
) -> None:
    """
    原位替换列内容（整列覆盖）：
    - 对已有列 col_name：逐行计算新值并覆盖
    - 若列不存在：由 if_missing 控制（error/skip/create）

    写出引号策略：
    - quote_mode=minimal：所有列按“最小必要引号”
    - quote_mode=newcol：将 col_name 视作“新增列”写出（受 col_value_type 控制是否强制双引号）
    """
    if if_missing not in {"error", "skip", "create"}:
        raise ValueError("if_missing 只能是: error / skip / create")
    if col_value_type not in {"auto", "str", "int", "float"}:
        raise ValueError("col_value_type 只能是: auto / str / int / float")
    if quote_mode not in {"minimal", "newcol"}:
        raise ValueError("quote_mode 只能是: minimal / newcol")

    # 先检查列是否存在（快速 fail / skip / create）
    col_exists = None
    try:
        cols0 = pd.read_csv(
            input_path,
            encoding=encoding,
            sep=delimiter,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            nrows=0,
        ).columns
        col_exists = col_name in cols0
        if not col_exists:
            if if_missing == "skip":
                os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
                shutil.copyfile(input_path, output_path)
                return
            if if_missing == "error":
                raise ValueError(f"列不存在: {col_name}")
            # create：允许后续流程创建该列
    except Exception as e:
        if "列不存在" in str(e):
            raise
        # 解析失败就走常规流程，让后续读/写给出更明确的错误
        pass

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    quotechar = '"'
    lineterminator = "\n"
    wrote_header = False

    # 是否将该列视作“新增列”强制引号
    new_col_names = [col_name] if quote_mode == "newcol" else []

    with open(output_path, "w", encoding=encoding, newline="") as f_out:
        for chunk in _iter_chunks(input_path, encoding=encoding, delimiter=delimiter, chunksize=chunksize):
            chunk = chunk.copy()

            if col_name not in chunk.columns:
                if if_missing == "create":
                    chunk[col_name] = _compute_new_col(chunk, value=value, from_cols=cols, sep=sep, template=template)
                elif if_missing == "skip":
                    # 列缺失：不改动（但会标准化写出格式）
                    pass
                else:
                    raise ValueError(f"列不存在: {col_name}")
            else:
                chunk[col_name] = _compute_new_col(chunk, value=value, from_cols=cols, sep=sep, template=template)

            # 写表头
            if not wrote_header:
                header_cells = list(chunk.columns)
                header_line = delimiter.join(
                    _format_csv_field(
                        str(h),
                        delimiter=delimiter,
                        quotechar=quotechar,
                        always_quote=False,
                        minimal_quote=True,
                    )
                    for h in header_cells
                )
                f_out.write(header_line + lineterminator)
                wrote_header = True

            # 写数据行
            col_names = list(chunk.columns)
            for tup in chunk.itertuples(index=False, name=None):
                interleaved: List[str] = []
                for c, v in zip(col_names, tup):
                    interleaved.append(c)
                    interleaved.append("" if v is None else str(v))
                _write_csv_row(
                    f_out,
                    interleaved,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    lineterminator=lineterminator,
                    new_col_names=new_col_names,
                    new_col_value_type=col_value_type,
                )


def _split_pipeline_steps(tokens: List[str]) -> List[List[str]]:
    """
    将 pipeline remainder tokens 切分成多个 step，每个 step 形如：
      ["addcol", "--new-col", "x", "--value", "y"]
    规则：遇到新的 op 名（_PIPE_OPS）即开始新 step。
    """
    steps: List[List[str]] = []
    cur: List[str] = []
    for t in tokens:
        if t in _PIPE_OPS:
            if cur:
                steps.append(cur)
            cur = [t]
        else:
            if not cur:
                raise ValueError(f"pipeline 语法错误：缺少子命令名（必须以 {_PIPE_OPS} 之一开头）；收到 token={t!r}")
            cur.append(t)
    if cur:
        steps.append(cur)
    return steps


def _make_step_parser(op: str) -> argparse.ArgumentParser:
    """
    为 pipeline 中的单步创建一个最小 argparse 解析器。
    注意：只解析该步自己的参数；全局 input/output/encoding 等由主命令控制。
    """
    p = argparse.ArgumentParser(prog=f"pipeline:{op}", add_help=False)
    if op == "addcol":
        p.add_argument("--new-col", required=True)
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--value")
        src.add_argument("--template")
        src.add_argument("--cols", nargs="+")
        p.add_argument("--sep", default="_")
        p.add_argument("--if-exists", default=None, choices=["error", "replace", "skip"])
        p.add_argument("--newcol-type", default=None, choices=["auto", "str", "int", "float"])
        return p
    if op == "mergecols":
        p.add_argument("--new-col", required=True)
        p.add_argument("--cols", nargs="+", required=True)
        p.add_argument("--sep", default="_")
        p.add_argument("--template", default=None)
        p.add_argument("--drop", action="store_true")
        p.add_argument("--if-exists", default=None, choices=["error", "replace", "skip"])
        p.add_argument("--newcol-type", default=None, choices=["auto", "str", "int", "float"])
        return p
    if op == "setcol":
        p.add_argument("--col", required=True)
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--value")
        src.add_argument("--template")
        src.add_argument("--cols", nargs="+")
        p.add_argument("--sep", default="_")
        p.add_argument("--if-missing", default="error", choices=["error", "skip", "create"])
        p.add_argument("--quote-mode", default="minimal", choices=["minimal", "newcol"])
        p.add_argument("--col-type", default=None, choices=["auto", "str", "int", "float"])
        return p
    if op == "addindex":
        p.add_argument("--new-col", default="index")
        p.add_argument("--seed", default="00000")
        p.add_argument("--step", type=int, default=1)
        p.add_argument("--pos", default="first", choices=["first", "last"])
        p.add_argument("--if-exists", default=None, choices=["error", "replace", "skip"])
        return p
    if op == "dropcol":
        p.add_argument("--col", required=True)
        p.add_argument("--if-exists", default="error", choices=["error", "skip"])
        return p
    if op == "replacepath":
        p.add_argument("--col", default="image_path")
        p.add_argument("--old", required=True)
        p.add_argument("--new", required=True)
        p.add_argument("--regex", action="store_true")
        p.add_argument("--if-missing", default="error", choices=["error", "skip"])
        return p
    if op == "unifyeditplan":
        p.add_argument("--src-col", required=True)
        p.add_argument("--new-col", required=True)
        p.add_argument("--if-src-missing", default="error", choices=["error", "skip"])
        p.add_argument("--if-exists", default=None, choices=["error", "replace", "skip"])
        return p
    if op == "extractcols":
        p.add_argument("--cols", nargs="+", default=None, help="要抽取的列名列表（按给定顺序写出）")
        p.add_argument(
            "--map",
            action="append",
            default=None,
            help="重命名映射 old:new 或 old=new；可重复传入。若未给 --cols，则按 --map 的 key 推断抽取列顺序。",
        )
        p.add_argument("--if-missing", default="error", choices=["error", "skip"])
        return p
    if op == "extractobjlist":
        p.add_argument("--src-col", required=True)
        p.add_argument("--new-col", required=True)
        p.add_argument("--dedupe", dest="dedupe", action="store_true", default=True)
        p.add_argument("--no-dedupe", dest="dedupe", action="store_false")
        p.add_argument("--if-src-missing", default="error", choices=["error", "skip"])
        p.add_argument("--if-exists", default=None, choices=["error", "replace", "skip"])
        return p
    raise ValueError(f"未知 pipeline op: {op}")


def run_pipeline(
    *,
    input_path: str,
    output_path: str,
    inplace: bool,
    steps_tokens: List[str],
    encoding: str,
    delimiter: str,
    chunksize: int,
    default_if_exists: str,
    default_newcol_type: str,
) -> None:
    """
    pipeline/multi：顺序执行多个步骤。
    - inplace=True：每一步 output 写到临时文件，然后原子替换回 input_path
    - inplace=False：中间步骤落到临时文件，最后一步写到 output_path
    """
    steps = _split_pipeline_steps(list(steps_tokens))
    if not steps:
        raise ValueError("pipeline 至少需要一个 step")

    # 当前输入文件
    cur_in = input_path

    # 临时文件放在 input 所在目录，保证 os.replace 跨设备不出问题
    base_dir = os.path.dirname(os.path.abspath(input_path)) or "."

    def _tmp_path() -> str:
        fd, p = tempfile.mkstemp(prefix=".__easy_act_pipe__", suffix=".csv", dir=base_dir)
        os.close(fd)
        return p

    for si, step in enumerate(steps):
        op = step[0]
        parser = _make_step_parser(op)
        ns = parser.parse_args(step[1:])

        is_last = si == len(steps) - 1
        if inplace:
            cur_out = _tmp_path()
        else:
            cur_out = output_path if is_last else _tmp_path()

        try:
            if op in ("addcol", "mergecols"):
                if_exists = ns.if_exists or default_if_exists
                newcol_type = ns.newcol_type or default_newcol_type
                process_csv(
                    input_path=cur_in,
                    output_path=cur_out,
                    op="addcol" if op == "addcol" else "mergecols",
                    new_col=ns.new_col,
                    value=getattr(ns, "value", None),
                    cols=getattr(ns, "cols", None),
                    sep=getattr(ns, "sep", "_"),
                    template=getattr(ns, "template", None),
                    drop=bool(getattr(ns, "drop", False)),
                    if_exists=if_exists,
                    encoding=encoding,
                    delimiter=delimiter,
                    chunksize=chunksize,
                    new_col_value_type=newcol_type,
                )
            elif op == "addindex":
                if_exists = ns.if_exists or default_if_exists
                add_index_to_csv(
                    input_path=cur_in,
                    output_path=cur_out,
                    index_col=ns.new_col,
                    seed=ns.seed,
                    step=ns.step,
                    pos=ns.pos,
                    if_exists=if_exists,
                    encoding=encoding,
                    delimiter=delimiter,
                    chunksize=chunksize,
                )
            elif op == "dropcol":
                drop_column_from_csv(
                    input_path=cur_in,
                    output_path=cur_out,
                    col_name=ns.col,
                    if_exists=ns.if_exists,
                    encoding=encoding,
                    delimiter=delimiter,
                    chunksize=chunksize,
                )
            elif op == "replacepath":
                replace_text_in_column_from_csv(
                    input_path=cur_in,
                    output_path=cur_out,
                    col_name=ns.col,
                    old=ns.old,
                    new=ns.new,
                    regex=bool(ns.regex),
                    if_missing=ns.if_missing,
                    encoding=encoding,
                    delimiter=delimiter,
                    chunksize=chunksize,
                )
            elif op == "unifyeditplan":
                if_exists = ns.if_exists or default_if_exists
                unify_editplan_column_in_csv(
                    input_path=cur_in,
                    output_path=cur_out,
                    src_col=ns.src_col,
                    new_col=ns.new_col,
                    if_src_missing=ns.if_src_missing,
                    if_exists=if_exists,
                    encoding=encoding,
                    delimiter=delimiter,
                    chunksize=chunksize,
                )
            elif op == "setcol":
                col_type = ns.col_type or default_newcol_type
                set_column_in_csv(
                    input_path=cur_in,
                    output_path=cur_out,
                    col_name=ns.col,
                    value=getattr(ns, "value", None),
                    cols=getattr(ns, "cols", None),
                    sep=getattr(ns, "sep", "_"),
                    template=getattr(ns, "template", None),
                    if_missing=ns.if_missing,
                    encoding=encoding,
                    delimiter=delimiter,
                    chunksize=chunksize,
                    col_value_type=col_type,
                    quote_mode=ns.quote_mode,
                )
            elif op == "extractcols":
                extract_columns_to_new_csv(
                    input_path=cur_in,
                    output_path=cur_out,
                    cols=getattr(ns, "cols", None),
                    rename_maps=getattr(ns, "map", None),
                    if_missing=getattr(ns, "if_missing", "error"),
                    encoding=encoding,
                    delimiter=delimiter,
                    chunksize=chunksize,
                )
            elif op == "extractobjlist":
                if_exists = ns.if_exists or default_if_exists
                extract_object_list_from_editplan_in_csv(
                    input_path=cur_in,
                    output_path=cur_out,
                    src_col=ns.src_col,
                    new_col=ns.new_col,
                    dedupe=bool(getattr(ns, "dedupe", True)),
                    if_src_missing=ns.if_src_missing,
                    if_exists=if_exists,
                    encoding=encoding,
                    delimiter=delimiter,
                    chunksize=chunksize,
                )
            else:
                raise ValueError(f"未知 op: {op}")

            if inplace:
                os.replace(cur_out, cur_in)
            else:
                if not is_last:
                    # 下一步以该临时文件作为输入
                    cur_in = cur_out
        finally:
            # 非最后一步：若 inplace 则 os.replace 已消费临时文件；若非 inplace，临时文件要保留作为下一步输入
            if inplace:
                try:
                    if os.path.exists(cur_out):
                        os.remove(cur_out)
                except Exception:
                    pass
            else:
                if is_last:
                    try:
                        if os.path.exists(cur_out) and cur_out != output_path:
                            os.remove(cur_out)
                    except Exception:
                        pass


def replace_text_in_column_from_csv(
    *,
    input_path: str,
    output_path: str,
    col_name: str = "image_path",
    old: str,
    new: str,
    regex: bool = False,
    if_missing: str = "error",  # error|skip
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
    chunksize: int = 200_000,
) -> None:
    """
    在指定列中做字符串替换（面向 path 场景，支持子串/正则）。

    注意：
    - 这是“改列内容”的操作，不属于“新增列”，因此写出时对该列也采用“最小必要引号”策略。
    - pandas 的 str.replace 在 regex=False 时执行字面量替换；regex=True 时为正则替换。
    """
    if if_missing not in {"error", "skip"}:
        raise ValueError("if_missing 只能是: error / skip")

    # 先检查列是否存在（快速 fail）
    try:
        cols0 = pd.read_csv(
            input_path,
            encoding=encoding,
            sep=delimiter,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            nrows=0,
        ).columns
        if col_name not in cols0:
            if if_missing == "skip":
                os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
                shutil.copyfile(input_path, output_path)
                return
            raise ValueError(f"列不存在: {col_name}")
    except Exception as e:
        if "列不存在" in str(e):
            raise
        # 解析失败就走常规流程，让后续读/写给出更明确的错误
        pass

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    quotechar = '"'
    lineterminator = "\n"
    wrote_header = False

    with open(output_path, "w", encoding=encoding, newline="") as f_out:
        for chunk in _iter_chunks(input_path, encoding=encoding, delimiter=delimiter, chunksize=chunksize):
            chunk = chunk.copy()
            if col_name not in chunk.columns:
                if if_missing == "skip":
                    # 列不存在：不做替换，直接写出当前 chunk（但写出格式会标准化）
                    pass
                else:
                    raise ValueError(f"列不存在: {col_name}")
            else:
                s = chunk[col_name].astype(str)
                chunk[col_name] = s.str.replace(old, new, regex=regex)

            # 写表头
            if not wrote_header:
                header_cells = list(chunk.columns)
                header_line = delimiter.join(
                    _format_csv_field(
                        str(h),
                        delimiter=delimiter,
                        quotechar=quotechar,
                        always_quote=False,
                        minimal_quote=True,
                    )
                    for h in header_cells
                )
                f_out.write(header_line + lineterminator)
                wrote_header = True

            # 写数据行：不涉及新增列 -> 所有列最小必要引号
            col_names = list(chunk.columns)
            for tup in chunk.itertuples(index=False, name=None):
                interleaved: List[str] = []
                for c, v in zip(col_names, tup):
                    interleaved.append(c)
                    interleaved.append("" if v is None else str(v))
                _write_csv_row(
                    f_out,
                    interleaved,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    lineterminator=lineterminator,
                    new_col_names=[],  # 不属于新增列
                    new_col_value_type="auto",
                )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CSV 数据处理工具（pandas 分块）：新增列、合并列、原位替换列内容、删除列；仅对指定列按需强制加双引号。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True, help="输入 CSV")

    out_group = p.add_mutually_exclusive_group(required=True)
    out_group.add_argument("-o", "--output", help="输出 CSV")
    out_group.add_argument("--inplace", action="store_true", help="原地覆盖写回输入文件")

    p.add_argument("--encoding", default="utf-8-sig", help="读写编码")
    p.add_argument("--delimiter", default=",", help="分隔符")
    p.add_argument("--chunksize", type=int, default=200_000, help="分块大小（行数）")
    p.add_argument(
        "--if-exists",
        default="error",
        choices=["error", "replace", "skip"],
        help="当目标列已存在时的处理方式",
    )
    p.add_argument(
        "--newcol-type",
        default="auto",
        choices=["auto", "str", "int", "float"],
        help="新增列的类型：决定是否强制双引号（str=永远；auto=像数字则不加）",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("addcol", help="新增一列")
    p_add.add_argument("--new-col", required=True, help="新增列名")
    src = p_add.add_mutually_exclusive_group(required=True)
    src.add_argument("--value", help="常量值（每行相同）")
    src.add_argument("--template", help='模板，如 "{a}_{b}"（占位符来自列名，逐行格式化，较慢）')
    src.add_argument("--cols", nargs="+", help="用于拼接的列名列表（配合 --sep）")
    p_add.add_argument("--sep", default="_", help="拼接分隔符（配合 --cols）")

    p_merge = sub.add_parser("mergecols", help="合并多列为新列")
    p_merge.add_argument("--new-col", required=True, help="新列名")
    p_merge.add_argument("--cols", nargs="+", required=True, help="要合并的列名列表")
    p_merge.add_argument("--sep", default="_", help="拼接分隔符")
    p_merge.add_argument("--template", default=None, help='模板，如 "{a}_{b}"（优先于 --sep）')
    p_merge.add_argument("--drop", action="store_true", help="合并后删除源列")

    p_set = sub.add_parser("setcol", help="原位替换列内容（整列覆盖）")
    p_set.add_argument("--col", required=True, help="要覆盖写入的列名（通常应已存在）")
    src = p_set.add_mutually_exclusive_group(required=True)
    src.add_argument("--value", help="常量值（每行相同）")
    src.add_argument("--template", help='模板，如 "{a}_{b}"（占位符来自列名，逐行格式化）')
    src.add_argument("--cols", nargs="+", help="用于拼接的列名列表（配合 --sep）")
    p_set.add_argument("--sep", default="_", help="拼接分隔符（配合 --cols）")
    p_set.add_argument(
        "--if-missing",
        default="error",
        choices=["error", "skip", "create"],
        help="当目标列不存在时的处理方式（error=报错，skip=直接复制，create=创建该列）",
    )
    p_set.add_argument(
        "--quote-mode",
        default="minimal",
        choices=["minimal", "newcol"],
        help="写出引号策略：minimal=所有列最小必要引号；newcol=将该列视作“新增列”并按类型决定是否强制双引号",
    )
    p_set.add_argument(
        "--col-type",
        default=None,
        choices=["auto", "str", "int", "float"],
        help="覆盖列的类型（仅在 --quote-mode newcol 时生效；默认继承全局 --newcol-type）",
    )

    p_idx = sub.add_parser("addindex", help="新增顺序 index 列（支持 seed 例如 00000）")
    p_idx.add_argument("--new-col", default="index", help="index 列名")
    p_idx.add_argument("--seed", default="00000", help='起始 seed（纯数字字符串），如 "00000" / "00012"')
    p_idx.add_argument("--step", type=int, default=1, help="步长")
    p_idx.add_argument("--pos", default="first", choices=["first", "last"], help="插入位置：first=第一列，last=最后一列")

    p_drop = sub.add_parser("dropcol", help="删除列")
    p_drop.add_argument("--col", required=True, help="要删除的列名")
    p_drop.add_argument(
        "--if-exists",
        default="error",
        choices=["error", "skip"],
        help="当列不存在时的处理方式（error=报错，skip=跳过）",
    )

    p_rp = sub.add_parser("replacepath", help="替换指定列（默认 image_path）中的部分内容（子串/正则）")
    p_rp.add_argument("--col", default="image_path", help="要替换的列名（默认 image_path）")
    p_rp.add_argument("--old", required=True, help="要被替换的字符串/正则模式")
    p_rp.add_argument("--new", required=True, help="替换成的字符串")
    p_rp.add_argument("--regex", action="store_true", help="将 --old 视为正则表达式（默认按字面量替换）")
    p_rp.add_argument(
        "--if-missing",
        default="error",
        choices=["error", "skip"],
        help="当列不存在时的处理方式（error=报错，skip=直接原样复制）",
    )

    p_u = sub.add_parser("unifyeditplan", help="统一两种 editplan JSON 格式为标准字段，并写入新列")
    p_u.add_argument("--src-col", required=True, help="源列名（内容为 JSON 列表字符串）")
    p_u.add_argument("--new-col", required=True, help="新列名（写入统一后的 JSON 列表字符串）")
    p_u.add_argument(
        "--if-src-missing",
        default="error",
        choices=["error", "skip"],
        help="源列不存在时处理：error=报错；skip=直接原样复制",
    )
    p_u.add_argument(
        "--if-exists",
        dest="if_exists_sub",
        default=None,
        choices=["error", "replace", "skip"],
        help="当新列已存在时的处理方式（若不填则继承全局 --if-exists）",
    )

    p_ext = sub.add_parser("extractcols", help="抽取指定列到新的 CSV（可重命名列名）")
    p_ext.add_argument("--cols", nargs="+", default=None, help="要抽取的列名列表（按给定顺序写出）")
    p_ext.add_argument(
        "--map",
        action="append",
        default=None,
        help="重命名映射 old:new 或 old=new；可重复传入。若未给 --cols，则按 --map 的 key 推断抽取列顺序。",
    )
    p_ext.add_argument(
        "--if-missing",
        default="error",
        choices=["error", "skip"],
        help="指定列缺失时处理：error=报错；skip=忽略缺失列（至少需剩 1 列）",
    )

    p_obj = sub.add_parser("extractobjlist", help="从标准字段 editplan JSON 中提取 object list 并写入新列")
    p_obj.add_argument("--src-col", required=True, help="源列名（内容为标准字段 editplan JSON 列表字符串）")
    p_obj.add_argument("--new-col", required=True, help="新列名（写入 object list，JSON 数组字符串）")
    p_obj.add_argument("--dedupe", dest="dedupe", action="store_true", default=True, help="对 object_name 去重（保持首次出现顺序）")
    p_obj.add_argument("--no-dedupe", dest="dedupe", action="store_false", help="不去重（保留原顺序中的重复项）")
    p_obj.add_argument(
        "--if-src-missing",
        default="error",
        choices=["error", "skip"],
        help="源列不存在时处理：error=报错；skip=直接原样复制",
    )
    p_obj.add_argument(
        "--if-exists",
        dest="if_exists_sub",
        default=None,
        choices=["error", "replace", "skip"],
        help="当新列已存在时的处理方式（若不填则继承全局 --if-exists）",
    )

    # pipeline / multi：将后续 tokens 作为“步骤串”解析
    for name in ("pipeline", "multi"):
        p_pipe = sub.add_parser(
            name,
            help="一次命令顺序执行多个子操作（addcol/mergecols/setcol/addindex/dropcol/replacepath/unifyeditplan/extractcols/extractobjlist）",
        )
        p_pipe.add_argument(
            "steps",
            nargs=argparse.REMAINDER,
            help="步骤串（每个步骤以子命令名开头，如: addcol ... mergecols ... replacepath ...）",
        )

    return p





def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.inplace:
        input_abs = os.path.abspath(args.input)
        dir_path = os.path.dirname(input_abs) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=".__easy_act__", suffix=".csv", dir=dir_path)
        os.close(fd)
        out_path = tmp_path
    else:
        out_path = args.output
        tmp_path = None
        input_abs = None

    try:
        if args.cmd == "addcol":
            process_csv(
                input_path=args.input,
                output_path=out_path,
                op="addcol",
                new_col=args.new_col,
                value=getattr(args, "value", None),
                cols=getattr(args, "cols", None),
                sep=getattr(args, "sep", "_"),
                template=getattr(args, "template", None),
                drop=False,
                if_exists=args.if_exists,
                encoding=args.encoding,
                delimiter=args.delimiter,
                chunksize=args.chunksize,
                new_col_value_type=args.newcol_type,
            )
        elif args.cmd == "mergecols":
            process_csv(
                input_path=args.input,
                output_path=out_path,
                op="mergecols",
                new_col=args.new_col,
                value=None,
                cols=args.cols,
                sep=args.sep,
                template=args.template,
                drop=args.drop,
                if_exists=args.if_exists,
                encoding=args.encoding,
                delimiter=args.delimiter,
                chunksize=args.chunksize,
                new_col_value_type=args.newcol_type,
            )
        elif args.cmd == "setcol":
            set_column_in_csv(
                input_path=args.input,
                output_path=out_path,
                col_name=args.col,
                value=getattr(args, "value", None),
                cols=getattr(args, "cols", None),
                sep=getattr(args, "sep", "_"),
                template=getattr(args, "template", None),
                if_missing=args.if_missing,
                encoding=args.encoding,
                delimiter=args.delimiter,
                chunksize=args.chunksize,
                col_value_type=(args.col_type or args.newcol_type),
                quote_mode=args.quote_mode,
            )
        elif args.cmd == "addindex":
            add_index_to_csv(
                input_path=args.input,
                output_path=out_path,
                index_col=args.new_col,
                seed=args.seed,
                step=args.step,
                pos=args.pos,
                if_exists=args.if_exists,
                encoding=args.encoding,
                delimiter=args.delimiter,
                chunksize=args.chunksize,
            )
        elif args.cmd == "dropcol":
            # dropcol 子命令有自己的 --if-exists 参数（含义：当列不存在时的处理方式）
            # argparse 中子命令的参数会覆盖主解析器的同名参数
            drop_column_from_csv(
                input_path=args.input,
                output_path=out_path,
                col_name=args.col,
                if_exists=args.if_exists,
                encoding=args.encoding,
                delimiter=args.delimiter,
                chunksize=args.chunksize,
            )
        elif args.cmd == "replacepath":
            replace_text_in_column_from_csv(
                input_path=args.input,
                output_path=out_path,
                col_name=args.col,
                old=args.old,
                new=args.new,
                regex=bool(args.regex),
                if_missing=args.if_missing,
                encoding=args.encoding,
                delimiter=args.delimiter,
                chunksize=args.chunksize,
            )
        elif args.cmd == "unifyeditplan":
            unify_editplan_column_in_csv(
                input_path=args.input,
                output_path=out_path,
                src_col=args.src_col,
                new_col=args.new_col,
                if_src_missing=args.if_src_missing,
                if_exists=(getattr(args, "if_exists_sub", None) or args.if_exists),
                encoding=args.encoding,
                delimiter=args.delimiter,
                chunksize=args.chunksize,
            )
        elif args.cmd == "extractcols":
            extract_columns_to_new_csv(
                input_path=args.input,
                output_path=out_path,
                cols=getattr(args, "cols", None),
                rename_maps=getattr(args, "map", None),
                if_missing=getattr(args, "if_missing", "error"),
                encoding=args.encoding,
                delimiter=args.delimiter,
                chunksize=args.chunksize,
            )
        elif args.cmd == "extractobjlist":
            extract_object_list_from_editplan_in_csv(
                input_path=args.input,
                output_path=out_path,
                src_col=args.src_col,
                new_col=args.new_col,
                dedupe=bool(getattr(args, "dedupe", True)),
                if_src_missing=args.if_src_missing,
                if_exists=(getattr(args, "if_exists_sub", None) or args.if_exists),
                encoding=args.encoding,
                delimiter=args.delimiter,
                chunksize=args.chunksize,
            )
        elif args.cmd in ("pipeline", "multi"):
            if not args.steps:
                raise ValueError("pipeline/multi 需要至少一个 step，例如：pipeline addcol ... mergecols ...")
            if args.inplace:
                run_pipeline(
                    input_path=args.input,
                    output_path=args.input,
                    inplace=True,
                    steps_tokens=list(args.steps),
                    encoding=args.encoding,
                    delimiter=args.delimiter,
                    chunksize=args.chunksize,
                    default_if_exists=args.if_exists,
                    default_newcol_type=args.newcol_type,
                )
            else:
                run_pipeline(
                    input_path=args.input,
                    output_path=out_path,
                    inplace=False,
                    steps_tokens=list(args.steps),
                    encoding=args.encoding,
                    delimiter=args.delimiter,
                    chunksize=args.chunksize,
                    default_if_exists=args.if_exists,
                    default_newcol_type=args.newcol_type,
                )
        else:
            raise ValueError(f"未知命令: {args.cmd}")

        if args.inplace and args.cmd not in ("pipeline", "multi"):
            os.replace(out_path, input_abs)  # type: ignore[arg-type]
    finally:
        if tmp_path:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

