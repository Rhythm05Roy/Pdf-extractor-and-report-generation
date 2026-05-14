"""
main.py — CLI entrypoint for LegalMind.

Usage:
  python main.py --file path/to/doc.pdf
  python main.py --file path/to/doc.pdf --query "breach of contract claim"
  python main.py --demo          # run on bundled sample documents
  python main.py --evaluate      # run evaluation suite
"""
import sys
import json
import argparse
from pathlib import Path
from loguru import logger
import config

# Configure logger
logger.remove()
logger.add(sys.stderr, level=config.LOG_LEVEL, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")


def run_pipeline(file_path: str, query: str = None, output_dir: str = None) -> dict:
    """
    Run the full LegalMind pipeline on a single document.

    Returns:
        dict with extracted_doc, structured_doc, draft_memo (as dicts)
    """
    from ingestion import load_document
    from extraction import extract_fields
    from retrieval import chunk_document, get_retriever
    from generation import generate_draft

    logger.info(f"{'='*60}")
    logger.info(f"Processing: {Path(file_path).name}")
    logger.info(f"{'='*60}")

    # Stage 1: Ingestion
    logger.info("[1/4] Ingesting document...")
    extracted_doc = load_document(file_path)
    if extracted_doc.ingestion_warnings:
        for w in extracted_doc.ingestion_warnings:
            logger.warning(f"  ⚠ {w}")

    # Stage 2: Extraction
    logger.info("[2/4] Extracting structured fields...")
    structured_doc = extract_fields(extracted_doc)
    logger.info(
        f"  Type: {structured_doc.document_type.value} "
        f"({structured_doc.document_type_confidence:.0%} conf)"
    )
    if structured_doc.case_number:
        logger.info(f"  Case: {structured_doc.case_number}")
    if structured_doc.parties:
        logger.info(f"  Parties: {', '.join(structured_doc.parties[:3])}")

    # Stage 3: Indexing
    logger.info("[3/4] Chunking and indexing for retrieval...")
    chunks = chunk_document(extracted_doc)
    retriever = get_retriever()
    retriever.index_chunks(chunks)
    logger.info(f"  Indexed {len(chunks)} chunks")

    # Stage 4: Draft generation
    logger.info("[4/4] Generating grounded draft memo...")
    draft = generate_draft(structured_doc, retriever, query=query)

    # Output
    output_dir = output_dir or "sample_outputs"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    doc_stem = Path(file_path).stem
    md_path = Path(output_dir) / f"{doc_stem}_draft.md"
    json_path = Path(output_dir) / f"{doc_stem}_structured.json"

    md_path.write_text(draft.to_markdown(), encoding="utf-8")
    json_path.write_text(json.dumps(structured_doc.to_dict(), indent=2), encoding="utf-8")

    logger.info(f"\n✅ Done!")
    logger.info(f"   Draft memo  → {md_path}")
    logger.info(f"   Structured  → {json_path}")
    logger.info(f"   Evidence chunks used: {draft.total_evidence_chunks}")
    logger.info(f"   Model: {draft.model_used}")

    return {
        "extracted": {
            "doc_id": extracted_doc.doc_id,
            "pages": extracted_doc.total_pages,
            "chars": len(extracted_doc.full_text),
            "avg_ocr_confidence": extracted_doc.avg_ocr_confidence,
        },
        "structured": structured_doc.to_dict(),
        "draft": draft.to_dict(),
        "output_files": {
            "markdown": str(md_path),
            "json": str(json_path),
        },
    }


def run_demo():
    """Run the pipeline on all sample documents."""
    sample_dir = config.SAMPLE_INPUTS_DIR
    docs = list(sample_dir.glob("*.pdf")) + list(sample_dir.glob("*.txt"))

    if not docs:
        logger.error(f"No sample documents found in {sample_dir}")
        logger.info("Run 'python scripts/generate_samples.py' to create sample documents")
        return

    logger.info(f"Running demo on {len(docs)} sample document(s)...")
    for doc_path in docs:
        try:
            run_pipeline(str(doc_path))
        except Exception as e:
            logger.error(f"Failed on {doc_path.name}: {e}")


def run_evaluation():
    """Run the evaluation suite."""
    from tests.evaluation import run_full_evaluation
    run_full_evaluation()


def main():
    parser = argparse.ArgumentParser(
        description="LegalMind — AI-powered legal document workflow for Pearson Specter Litt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --file data/sample_inputs/contract.pdf
  python main.py --file data/sample_inputs/notice.txt --query "breach notice deadline"
  python main.py --demo
  python main.py --evaluate
        """,
    )
    parser.add_argument("--file", type=str, help="Path to document to process")
    parser.add_argument("--query", type=str, help="Custom retrieval query (optional)")
    parser.add_argument("--output", type=str, default="sample_outputs", help="Output directory")
    parser.add_argument("--demo", action="store_true", help="Run on bundled sample documents")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation suite")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit UI")

    args = parser.parse_args()

    if args.ui:
        import subprocess
        subprocess.run(["streamlit", "run", "app.py"])
        return

    if args.demo:
        run_demo()
        return

    if args.evaluate:
        run_evaluation()
        return

    if args.file:
        result = run_pipeline(args.file, query=args.query, output_dir=args.output)
        print("\n" + "="*60)
        print("PIPELINE RESULT SUMMARY")
        print("="*60)
        print(f"Document ID:    {result['extracted']['doc_id'][:8]}...")
        print(f"Pages:          {result['extracted']['pages']}")
        print(f"Characters:     {result['extracted']['chars']:,}")
        print(f"Doc Type:       {result['structured']['document_type']}")
        print(f"Case Number:    {result['structured']['case_number'] or 'N/A'}")
        print(f"Evidence Used:  {result['draft'].get('total_evidence_chunks', result['draft'].get('sections', []))} chunks")
        print(f"Model:          {result['draft']['model_used']}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
