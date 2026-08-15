# API reference

## Public lifecycle

::: ofiqpy.assessor.Assessor

::: ofiqpy.assess_typed

::: ofiqpy.assess

## Typed results

::: ofiqpy.results.AssessmentResult

::: ofiqpy.results.ComponentResult

::: ofiqpy.results.AssessmentStatus

::: ofiqpy.results.ComponentStatus

::: ofiqpy.results.FailureCode

## Verified profile and configuration

::: ofiqpy.profile.CanonicalProfile

::: ofiqpy.config.OFIQConfig

## Advanced pipeline access

`OFIQPipeline` and `Measures` expose internal preprocessing and execution stages. They do
not provide `Assessor`'s lock, array validation, or typed image-level failure boundary.

::: ofiqpy.pipeline.OFIQPipeline

::: ofiqpy.session.Session

::: ofiqpy.measures.core.Measures

## Output and batch

::: ofiqpy.output.header

::: ofiqpy.output.row

::: ofiqpy.batch.run_batch

## Conformance

::: ofiqpy.conformance.run_conformance
