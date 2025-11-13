# Refactoring Plan: Replace Custom Types with PyTorch Native Types and torchmetrics

## Overview

This PR will perform a major refactoring to:
1. Remove all custom types under `library/src/getiprompt/types/`
2. Replace `SegmentationMetrics` with `torchmetrics.MeanIoU`
3. Update all `nn.Module` components to use Python/PyTorch generic types

## PR Details

- **Branch Name**: `refactor/remove-custom-types-use-torchmetrics`
- **PR Title**: `refactor: replace custom types with standard PyTorch types and torchmetrics`
- **Type**: Breaking Change (Major Refactoring)

## Custom Types to Replace

| Custom Type | PyTorch Equivalent | Notes |
|-------------|-------------------|-------|
| `Masks` | `torch.Tensor` | Shape: `[N, H, W]` for N masks |
| `Points` | `torch.Tensor` | Shape: `[N, 2]` for N points |
| `Boxes` | `torch.Tensor` | Shape: `[N, 4]` for N boxes |
| `Results` | `dict[str, Any]` | Contains various outputs |

## Step-by-Step Implementation Plan

### Phase 1: Preparation and Analysis

#### 1.1 Create Type Mapping Documentation
- [ ] Document current usage patterns for each custom type
- [ ] Identify all components that consume/produce these types
- [ ] Map data flow through the system
- [ ] Create equivalence tests for current behavior

#### 1.2 Set Up Testing Infrastructure
- [ ] Create comprehensive test suite for current functionality
- [ ] Set up performance benchmarks
- [ ] Create regression test data
- [ ] Document expected behavior for each component

### Phase 2: Replace SegmentationMetrics

#### 2.1 Install Dependencies
- [ ] Add `torchmetrics` to `pyproject.toml`
- [ ] Update lock file
- [ ] Verify installation

#### 2.2 Create Metrics Adapter
- [ ] Create wrapper for `torchmetrics.MeanIoU` to match current interface
- [ ] Implement equivalent methods (`forward`, `get_metrics`)
- [ ] Ensure backward compatibility during transition

#### 2.3 Update Metrics Usage
- [ ] Update `library/src/getiprompt/scripts/benchmark.py`
- [ ] Update `library/tests/unit/scripts/test_benchmark.py`
- [ ] Verify metrics calculations produce equivalent results

#### 2.4 Test Metrics Replacement
- [ ] Run existing tests to ensure no regression
- [ ] Compare metrics outputs between old and new implementations
- [ ] Performance test to ensure no significant slowdown

### Phase 3: Component-by-Component Migration

#### 3.1 Leaf Components (No Dependencies)
Start with components that don't depend on other custom types:

- [ ] **`CosineSimilarity`**
  - Replace input/output types with `torch.Tensor`
  - Update method signatures
  - Update tests

- [ ] **`ImageEncoder`**
  - Replace `Image` input with `torch.Tensor`
  - Replace `Features` output with `torch.Tensor`
  - Update method signatures
  - Update tests

- [ ] **`MasksToPolygons`**
  - Replace `Masks` input with `torch.Tensor`
  - Update output format
  - Update method signatures
  - Update tests

#### 3.2 Intermediate Components
Components that depend on leaf components:

- [ ] **`MaskAdder`**
  - Update to work with `torch.Tensor` masks
  - Update `Priors` handling
  - Update method signatures
  - Update tests

- [ ] **`SamDecoder`**
  - Update input/output types
  - Update method signatures
  - Update tests

- [ ] **Feature Selectors**
  - `AverageFeatures`: Update to work with `torch.Tensor`
  - `ClusterFeatures`: Update to work with `torch.Tensor`
  - Update method signatures
  - Update tests

#### 3.3 Complex Components
Components that orchestrate multiple sub-components:

- [ ] **Prompt Generators**
  - `BidirectionalPromptGenerator`: Update all type usage
  - `SoftmatcherPromptGenerator`: Update all type usage
  - `GridPromptGenerator`: Update all type usage
  - `TextToBoxPromptGenerator`: Update all type usage
  - Update method signatures
  - Update tests

- [ ] **Grounding Model**
  - Update type usage
  - Update method signatures
  - Update tests

### Phase 4: Update Data Flow and Interfaces

#### 4.1 Model Interfaces
- [ ] Update `Model.learn()` method signature
- [ ] Update `Model.infer()` method signature
- [ ] Update model base classes
- [ ] Update all model implementations

#### 4.2 Dataset Interfaces
- [ ] Update dataset classes to work with native types
- [ ] Update data loading and collation
- [ ] Update sample conversion utilities
- [ ] Update dataset tests

#### 4.3 Visualization Components
- [ ] Update `ExportMaskVisualization`
- [ ] Update visualization utilities
- [ ] Update visualization tests

#### 4.4 CLI and Scripts
- [ ] Update CLI interfaces
- [ ] Update benchmark script
- [ ] Update other utility scripts
- [ ] Update script tests

### Phase 5: Cleanup and Finalization

#### 5.1 Remove Custom Types
- [ ] Delete `library/src/getiprompt/types/` directory
- [ ] Remove all imports of custom types
- [ ] Update `__init__.py` files
- [ ] Remove type-related documentation

#### 5.2 Update Documentation
- [ ] Update API documentation
- [ ] Update examples and tutorials
- [ ] Update README files
- [ ] Create migration guide for users

#### 5.3 Final Testing
- [ ] Run full test suite
- [ ] Run performance benchmarks
- [ ] Test with existing datasets
- [ ] Verify no regressions

## Testing Strategy

### Unit Tests
- [ ] Test each component individually with new types
- [ ] Verify input/output behavior matches expectations
- [ ] Test edge cases and error conditions

### Integration Tests
- [ ] Test complete pipelines with new types
- [ ] Verify end-to-end functionality
- [ ] Test with real datasets

### Regression Tests
- [ ] Compare outputs before/after refactoring
- [ ] Verify metrics calculations are equivalent
- [ ] Performance regression testing

### Migration Tests
- [ ] Test data conversion utilities
- [ ] Test backward compatibility during transition
- [ ] Test error handling for invalid inputs

## Risk Mitigation

### Breaking Changes
- **Risk**: All existing code using custom types will break
- **Mitigation**: 
  - Provide clear migration guide
  - Implement deprecation warnings before removal
  - Update all examples and documentation

### Data Compatibility
- **Risk**: Serialized models/data may become incompatible
- **Mitigation**:
  - Create data migration utilities
  - Document breaking changes clearly
  - Provide conversion scripts

### Performance Impact
- **Risk**: New implementation may be slower
- **Mitigation**:
  - Benchmark before/after performance
  - Optimize critical paths
  - Profile memory usage

## Success Criteria

- [ ] All custom types removed from codebase
- [ ] All components use PyTorch native types
- [ ] `torchmetrics.MeanIoU` replaces `SegmentationMetrics`
- [ ] All tests pass
- [ ] No performance regression
- [ ] Documentation updated
- [ ] Migration guide provided

## Rollback Plan

If issues arise during implementation:

1. **Immediate**: Revert to previous commit
2. **Partial**: Keep custom types alongside new implementation
3. **Gradual**: Implement feature flags for gradual migration

## Timeline Estimate

- **Phase 1**: 1-2 days (Preparation)
- **Phase 2**: 2-3 days (Metrics replacement)
- **Phase 3**: 5-7 days (Component migration)
- **Phase 4**: 3-4 days (Interface updates)
- **Phase 5**: 2-3 days (Cleanup)

**Total Estimated Time**: 2-3 weeks

## Dependencies

- `torchmetrics` package
- Updated test data
- Migration utilities
- Documentation updates

## Notes

- This is a major breaking change
- All users will need to update their code
- Consider version bump to indicate breaking changes
- Plan for comprehensive testing and validation
- Consider staging the changes in multiple smaller PRs if needed
