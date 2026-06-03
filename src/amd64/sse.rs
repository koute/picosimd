use core::arch::x86_64::*;

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct si128(pub __m128i);

impl si128 {
    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn zero() -> Self {
        maybe_unsafe! { Self(_mm_setzero_si128()) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn negative_one() -> Self {
        i32x4::splat(-1).as_si128()
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn from_i64_zext(value: i64) -> Self {
        maybe_unsafe! { Self(_mm_cvtsi64_si128(value)) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn with_upper_i64_clear(self) -> Self {
        si128::from_i64_zext(self.lo())
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub unsafe fn store_unaligned(self, address: *mut u8) {
        unsafe { _mm_storeu_si128(address.cast(), self.0) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub unsafe fn store_aligned(self, address: *mut u8) {
        unsafe { _mm_store_si128(address.cast(), self.0) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub unsafe fn load_unaligned(address: *const u8) -> Self {
        unsafe { Self(_mm_loadu_si128(address.cast())) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub unsafe fn load_aligned(address: *const u8) -> Self {
        unsafe { Self(_mm_load_si128(address.cast())) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn from_bytes(xs: [u8; 16]) -> Self {
        maybe_unsafe! {
            Self(_mm_setr_epi8(
                xs[0] as i8,
                xs[1] as i8,
                xs[2] as i8,
                xs[3] as i8,
                xs[4] as i8,
                xs[5] as i8,
                xs[6] as i8,
                xs[7] as i8,
                xs[8] as i8,
                xs[9] as i8,
                xs[10] as i8,
                xs[11] as i8,
                xs[12] as i8,
                xs[13] as i8,
                xs[14] as i8,
                xs[15] as i8,
            ))
        }
    }

    #[inline]
    pub fn as_i8x16(self) -> i8x16 {
        i8x16(self.0)
    }

    #[inline]
    pub fn as_i16x8(self) -> i16x8 {
        i16x8(self.0)
    }

    #[inline]
    pub fn as_i32x4(self) -> i32x4 {
        i32x4(self.0)
    }

    #[inline]
    pub fn as_i64x2(self) -> i64x2 {
        i64x2(self.0)
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn unbounded_shl_by_bytes<const AMOUNT: i32>(self) -> Self {
        maybe_unsafe! { Self(_mm_slli_si128(self.0, AMOUNT)) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn unbounded_shr_by_bytes<const AMOUNT: i32>(self) -> Self {
        maybe_unsafe! { Self(_mm_srli_si128(self.0, AMOUNT)) }
    }

    #[target_feature(enable = "sse4.1")]
    #[inline]
    pub fn is_equal(self, rhs: Self) -> bool {
        let cmp = self.as_i32x4().simd_eq(rhs.as_i32x4());
        maybe_unsafe! { _mm_testc_si128(cmp.0, i32x4::splat(-1).0) != 0 }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn is_equal_slow(self, rhs: Self) -> bool {
        self.as_i32x4().simd_eq(rhs.as_i32x4()).as_i8x16().most_significant_bits() == -1
    }

    #[target_feature(enable = "sse4.1")]
    #[inline]
    pub fn is_zero(self) -> bool {
        maybe_unsafe! { _mm_testz_si128(self.0, self.0) != 0 }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn and_not(self, rhs: Self) -> Self {
        maybe_unsafe! { Self(_mm_andnot_si128(rhs.0, self.0)) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn and(self, rhs: Self) -> Self {
        maybe_unsafe! { Self(_mm_and_si128(self.0, rhs.0)) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn or(self, rhs: Self) -> Self {
        maybe_unsafe! { Self(_mm_or_si128(self.0, rhs.0)) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn xor(self, rhs: Self) -> Self {
        maybe_unsafe! { Self(_mm_xor_si128(self.0, rhs.0)) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn lo(self) -> i64 {
        self.as_i64x2().lo()
    }

    #[target_feature(enable = "sse4.1")]
    #[inline]
    pub fn hi(self) -> i64 {
        self.as_i64x2().hi()
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn swap_halves(self) -> Self {
        crate::amd64::sse::i32x4_pick!(self, [2, 3, 0, 1]).as_si128()
    }

    #[target_feature(enable = "sse")]
    #[inline]
    pub fn copy_hi_to_lo(self, rhs: Self) -> Self {
        #[inline(always)]
        fn to_ps(value: __m128i) -> __m128 {
            unsafe { core::mem::transmute(value) }
        }

        #[inline(always)]
        fn from_ps(value: __m128) -> __m128i {
            unsafe { core::mem::transmute(value) }
        }

        maybe_unsafe! { Self(from_ps(_mm_movehl_ps(to_ps(self.0), to_ps(rhs.0)))) }
    }
}

impl_bitops!(si128);

macro_rules! impl_m128 {
    (
        $type:ident
        $add:ident
        $sub:ident
        $cmpeq:ident
        $cmpgt:ident
        $lane_ty:ident
        $lane_count:expr,
        $eq_target_feature:expr
    ) => {
        #[repr(transparent)]
        #[derive(Copy, Clone)]
        pub struct $type(pub __m128i);

        impl $type {
            #[target_feature(enable = "sse2")]
            #[inline]
            pub fn zero() -> Self {
                Self(si128::zero().0)
            }

            #[target_feature(enable = "sse2")]
            #[inline]
            pub fn negative_one() -> Self {
                Self(si128::negative_one().0)
            }

            #[inline]
            pub fn as_si128(self) -> si128 {
                si128(self.0)
            }

            #[inline]
            pub fn as_i8x16(self) -> i8x16 {
                self.as_si128().as_i8x16()
            }

            #[inline]
            pub fn as_i16x8(self) -> i16x8 {
                self.as_si128().as_i16x8()
            }

            #[inline]
            pub fn as_i32x4(self) -> i32x4 {
                self.as_si128().as_i32x4()
            }

            #[inline]
            pub fn as_i64x2(self) -> i64x2 {
                self.as_si128().as_i64x2()
            }

            #[inline]
            pub fn as_slice(&self) -> &[$lane_ty; $lane_count] {
                unsafe {
                    &*core::ptr::addr_of!(self.0).cast::<[$lane_ty; $lane_count]>()
                }
            }

            #[inline]
            pub fn as_slice_mut(&mut self) -> &mut [$lane_ty; $lane_count] {
                unsafe {
                    &mut *core::ptr::addr_of_mut!(self.0).cast::<[$lane_ty; $lane_count]>()
                }
            }

            #[target_feature(enable = "sse2")]
            #[inline]
            pub fn to_array(self) -> [$lane_ty; $lane_count] {
                let mut array: core::mem::MaybeUninit<[$lane_ty; $lane_count]> = core::mem::MaybeUninit::uninit();
                unsafe {
                    self.as_si128().store_unaligned(array.as_mut_ptr().cast());
                    array.assume_init()
                }
            }

            #[target_feature(enable = "sse2")]
            #[inline]
            pub unsafe fn load_unaligned(address: *const u8) -> Self {
                unsafe {
                    Self(si128::load_unaligned(address).0)
                }
            }

            #[target_feature(enable = $eq_target_feature)]
            #[inline]
            pub fn simd_eq(self, rhs: Self) -> Self {
                maybe_unsafe! { Self($cmpeq(self.0, rhs.0)) }
            }

            #[target_feature(enable = $eq_target_feature)]
            #[inline]
            pub fn simd_gt(self, rhs: Self) -> Self {
                maybe_unsafe! { Self($cmpgt(self.0, rhs.0)) }
            }

            #[target_feature(enable = $eq_target_feature)]
            #[inline]
            pub fn simd_lt(self, rhs: Self) -> Self {
                rhs.simd_gt(self)
            }

            #[target_feature(enable = "sse4.1")]
            #[inline]
            pub fn is_equal(self, rhs: Self) -> bool {
                self.as_si128().is_equal(rhs.as_si128())
            }

            #[target_feature(enable = "sse2")]
            #[inline]
            pub fn is_equal_slow(self, rhs: Self) -> bool {
                self.as_si128().is_equal_slow(rhs.as_si128())
            }

            #[target_feature(enable = "sse4.1")]
            #[inline]
            pub fn is_zero(self) -> bool {
                self.as_si128().is_zero()
            }

            #[target_feature(enable = "sse2")]
            #[inline]
            pub fn add(self, rhs: Self) -> Self {
                maybe_unsafe! { Self($add(self.0, rhs.0)) }
            }

            #[target_feature(enable = "sse2")]
            #[inline]
            pub fn sub(self, rhs: Self) -> Self {
                maybe_unsafe! { Self($sub(self.0, rhs.0)) }
            }
        }

        impl_bitops!($type as_si128 "sse2");
        impl_common_ops!($type);
    };
}

impl_m128!(i8x16 _mm_add_epi8 _mm_sub_epi8 _mm_cmpeq_epi8 _mm_cmpgt_epi8 i8 16, "sse2");
impl_m128!(i16x8 _mm_add_epi16 _mm_sub_epi16 _mm_cmpeq_epi16 _mm_cmpgt_epi16 i16 8, "sse2");
impl_m128!(i32x4 _mm_add_epi32 _mm_sub_epi32 _mm_cmpeq_epi32 _mm_cmpgt_epi32 i32 4, "sse2");
impl_m128!(i64x2 _mm_add_epi64 _mm_sub_epi64 _mm_cmpeq_epi64 _mm_cmpgt_epi64 i64 2, "sse4.2");

impl_min_max!(i8x16 _mm_min_epu8 _mm_max_epu8 _mm_min_epi8 _mm_max_epi8 "sse2", "sse4.1");
impl_min_max!(i16x8 _mm_min_epu16 _mm_max_epu16 _mm_min_epi16 _mm_max_epi16 "sse4.1", "sse2");
impl_min_max!(i32x4 _mm_min_epu32 _mm_max_epu32 _mm_min_epi32 _mm_max_epi32 "sse4.1", "sse4.1");

impl i8x16 {
    #[inline]
    pub const fn from_fallback(value: crate::fallback::i8x16) -> Self {
        unsafe { core::mem::transmute(value) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn splat(value: i8) -> Self {
        maybe_unsafe! { Self(_mm_set1_epi8(value)) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn from_array(xs: [i8; 16]) -> Self {
        let xs: [u8; 16] = unsafe { core::mem::transmute(xs) };
        si128::from_bytes(xs).as_i8x16()
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn lane_indexes() -> Self {
        Self::from_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn most_significant_bits(self) -> i16 {
        maybe_unsafe! { _mm_movemask_epi8(self.0) as i16 }
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    pub fn shuffle(self, mask: Self) -> Self {
        maybe_unsafe! {
            Self(_mm_shuffle_epi8(self.0, mask.0))
        }
    }

    #[must_use]
    #[target_feature(enable = "sse4.1")]
    #[inline]
    pub fn get<const INDEX: i32>(self) -> i8 {
        let value: i32 = maybe_unsafe! { _mm_extract_epi8(self.0, INDEX) };
        value as i8
    }

    #[must_use]
    #[target_feature(enable = "sse4.1")]
    #[inline]
    pub fn set<const INDEX: i32>(self, value: i32) -> Self {
        maybe_unsafe! { Self(_mm_insert_epi8(self.0, value, INDEX)) }
    }
}

impl i16x8 {
    #[inline]
    pub const fn from_fallback(value: crate::fallback::i16x8) -> Self {
        unsafe { core::mem::transmute(value) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn splat(value: i16) -> Self {
        maybe_unsafe! { Self(_mm_set1_epi16(value)) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn from_array(xs: [i16; 8]) -> Self {
        maybe_unsafe! {
            Self(_mm_setr_epi16(
                xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7],
            ))
        }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn clamp_to_i8_range_and_pack(self, rhs: Self) -> i8x16 {
        maybe_unsafe! {
            i8x16(_mm_packs_epi16(self.0, rhs.0))
        }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn clamp_to_u8_range_and_pack(self, rhs: Self) -> i8x16 {
        maybe_unsafe! {
            i8x16(_mm_packus_epi16(self.0, rhs.0))
        }
    }

    #[must_use]
    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn get<const INDEX: i32>(self) -> i16 {
        let value: i32 = maybe_unsafe! { _mm_extract_epi16(self.0, INDEX) };
        value as i16
    }

    #[must_use]
    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn set<const INDEX: i32>(self, value: i32) -> Self {
        maybe_unsafe! { Self(_mm_insert_epi16(self.0, value, INDEX)) }
    }
}

impl i32x4 {
    #[inline]
    pub const fn from_fallback(value: crate::fallback::i32x4) -> Self {
        unsafe { core::mem::transmute(value) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn splat(value: i32) -> Self {
        maybe_unsafe! { Self(_mm_set1_epi32(value)) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn from_array(xs: [i32; 4]) -> Self {
        maybe_unsafe! {
            Self(_mm_setr_epi32(
                xs[0], xs[1], xs[2], xs[3],
            ))
        }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn clamp_to_i16_range_and_pack(self, rhs: Self) -> i16x8 {
        maybe_unsafe! {
            i16x8(_mm_packs_epi32(self.0, rhs.0))
        }
    }

    #[target_feature(enable = "sse4.1")]
    #[inline]
    pub fn clamp_to_u16_range_and_pack(self, rhs: Self) -> i16x8 {
        maybe_unsafe! {
            i16x8(_mm_packus_epi32(self.0, rhs.0))
        }
    }

    #[must_use]
    #[target_feature(enable = "sse4.1")]
    #[inline]
    pub fn get<const INDEX: i32>(self) -> i32 {
        maybe_unsafe! { _mm_extract_epi32(self.0, INDEX) }
    }

    #[must_use]
    #[target_feature(enable = "sse4.1")]
    #[inline]
    pub fn set<const INDEX: i32>(self, value: i32) -> Self {
        maybe_unsafe! { Self(_mm_insert_epi32(self.0, value, INDEX)) }
    }

    #[doc(hidden)]
    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn _pick<const LANES: i32>(self) -> Self {
        assert_eq!(LANES & (!0xff), 0);
        maybe_unsafe! {
            Self(_mm_shuffle_epi32(self.0, LANES))
        }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn get_0th(self) -> i32 {
        maybe_unsafe! { _mm_cvtsi128_si32(self.0) }
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! _i32x4_pick {
    ($value:ident[$l0:expr, $l1:expr, $l2:expr, $l3:expr]) => {
        $crate::amd64::sse::i32x4_pick!($value, [$l0, $l1, $l2, $l3])
    };

    ($value:expr, [$l0:expr, $l1:expr, $l2:expr, $l3:expr]) => {{
        let _: () = {
            assert!($l0 < 8);
            assert!($l1 < 8);
            assert!($l2 < 8);
            assert!($l3 < 8);
        };

        let input: $crate::amd64::sse::i32x4 = $value.as_i32x4();
        input._pick::<{ (($l0) | ($l1 << 2) | ($l2 << 4) | ($l3 << 6)) & 0xff }>()
    }};
}

pub use _i32x4_pick as i32x4_pick;

impl i64x2 {
    #[inline]
    pub const fn from_fallback(value: crate::fallback::i64x2) -> Self {
        unsafe { core::mem::transmute(value) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn splat(value: i64) -> Self {
        maybe_unsafe! { Self(_mm_set1_epi64x(value)) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn from_array(xs: [i64; 2]) -> Self {
        maybe_unsafe! {
            Self(_mm_set_epi64x(
                xs[1], xs[0],
            ))
        }
    }

    #[must_use]
    #[target_feature(enable = "sse4.1")]
    #[inline]
    pub fn get<const INDEX: i32>(self) -> i64 {
        maybe_unsafe! { _mm_extract_epi64(self.0, INDEX) }
    }

    #[must_use]
    #[target_feature(enable = "sse4.1")]
    #[inline]
    pub fn set<const INDEX: i32>(self, value: i64) -> Self {
        maybe_unsafe! { Self(_mm_insert_epi64(self.0, value, INDEX)) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub fn lo(self) -> i64 {
        maybe_unsafe! { _mm_cvtsi128_si64(self.0) }
    }

    #[target_feature(enable = "sse4.1")]
    #[inline]
    pub fn hi(self) -> i64 {
        self.get::<1>()
    }
}

#[cfg(all(feature = "std", test))]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn basic() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }

        unsafe {
            assert_eq!(i32x4::from_array(test_array_i32x4()).get::<0>(), test_array_i32x4()[0]);
            assert_eq!(i32x4::from_array(test_array_i32x4()).to_array(), test_array_i32x4());
            assert_eq!(i64x2::from_array(test_array_i64x2()).get::<0>(), test_array_i64x2()[0]);
            assert_eq!(i64x2::from_array(test_array_i64x2()).to_array(), test_array_i64x2());
            assert_eq!(i64x2::from_array(test_array_i64x2()).lo(), test_array_i64x2()[0]);
            assert_eq!(i64x2::from_array(test_array_i64x2()).hi(), test_array_i64x2()[1]);

            assert!(i32x4::splat(0x12345678).is_equal(i32x4::splat(0x12345678)));
            assert!(!i32x4::splat(0x12345678).is_equal(i32x4::splat(0x12345678).set::<0>(0xff)));

            assert!(i32x4::splat(0x12345678).is_equal_slow(i32x4::splat(0x12345678)));
            assert!(!i32x4::splat(0x12345678).is_equal_slow(i32x4::splat(0x12345678).set::<0>(0xff)));

            assert!(!i8x16::zero().set::<0>(-1).is_equal(i8x16::zero()));

            assert!(i8x16::zero().is_zero());
            assert!(!i8x16::zero().set::<0>(-1).is_zero());
        }
    }

    #[test]
    fn i32x4_pick_macro() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }

        unsafe {
            let xs = test_array_i32x4();
            let ys = i32x4::from_array(xs);
            assert_eq!(ys.to_array(), xs);

            assert_eq!(i32x4_pick!(ys[0, 0, 0, 0]).to_array(), [xs[0], xs[0], xs[0], xs[0]]);
            assert_eq!(i32x4_pick!(ys[1, 1, 1, 1]).to_array(), [xs[1], xs[1], xs[1], xs[1]]);
            assert_eq!(i32x4_pick!(ys[0, 0, 3, 3]).to_array(), [xs[0], xs[0], xs[3], xs[3]]);
            assert_eq!(i32x4_pick!(ys[3, 3, 0, 0]).to_array(), [xs[3], xs[3], xs[0], xs[0]]);
            assert_eq!(i32x4_pick!(ys[3, 2, 1, 0]).to_array(), [xs[3], xs[2], xs[1], xs[0]]);
            assert_eq!(i32x4_pick!(ys[0, 3, 0, 3]).to_array(), [xs[0], xs[3], xs[0], xs[3]]);
            assert_eq!(i32x4_pick!(ys[3, 0, 3, 0]).to_array(), [xs[3], xs[0], xs[3], xs[0]]);
        }
    }

    #[test]
    fn si128_from_i64_zext() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        unsafe {
            let v = si128::from_i64_zext(0x1234567890abcdef);
            assert_eq!(v.lo(), 0x1234567890abcdef_i64);
            let cleared = v.with_upper_i64_clear();
            assert_eq!(cleared.lo(), 0x1234567890abcdef_i64);
            assert!(cleared.as_i64x2().get::<1>() == 0);
        }
    }

    #[test]
    fn si128_store_aligned_and_load_aligned() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        unsafe {
            #[repr(align(16))]
            struct Aligned16([u8; 16]);
            let mut aligned = Aligned16([0; 16]);
            let ptr = aligned.0.as_mut_ptr();
            let original = si128::from_bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
            original.store_aligned(ptr);
            let loaded = si128::load_aligned(ptr);
            assert_eq!(
                loaded.as_i8x16().to_array(),
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            );
        }
    }

    #[test]
    fn si128_unbounded_shl_by_bytes() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        unsafe {
            let bytes = si128::from_bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
            let shifted = bytes.unbounded_shl_by_bytes::<4>();
            assert_eq!(shifted.as_i8x16().to_array(), [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        }
    }

    #[test]
    fn si128_unbounded_shr_by_bytes() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        unsafe {
            let bytes = si128::from_bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
            let shifted = bytes.unbounded_shr_by_bytes::<4>();
            assert_eq!(
                shifted.as_i8x16().to_array(),
                [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0]
            );
        }
    }

    #[test]
    fn si128_swap_halves() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        unsafe {
            let bytes = si128::from_bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
            let swapped = bytes.swap_halves();
            assert_eq!(
                swapped.as_i8x16().to_array(),
                [9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8]
            );
        }
    }

    #[test]
    fn si128_copy_hi_to_lo() {
        if !std::arch::is_x86_feature_detected!("sse") {
            return;
        }
        unsafe {
            let a = si128::from_bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
            let b = si128::from_bytes([17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]);
            let copied = a.copy_hi_to_lo(b);
            assert_eq!(
                copied.as_i8x16().to_array(),
                [25, 26, 27, 28, 29, 30, 31, 32, 9, 10, 11, 12, 13, 14, 15, 16]
            );
        }
    }

    #[test]
    fn i8x16_most_significant_bits() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        unsafe {
            let v = i8x16::from_array([0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1]);
            assert_eq!(v.most_significant_bits(), 0b1010101010101010_u16 as i16);
        }
    }

    #[test]
    fn i8x16_shuffle() {
        if !std::arch::is_x86_feature_detected!("ssse3") {
            return;
        }
        unsafe {
            let v = i8x16::from_array([
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                110,
                120,
                130_u8 as i8,
                140_u8 as i8,
                150_u8 as i8,
                160_u8 as i8,
            ]);

            let mask = i8x16::from_array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
            let shuffled = v.shuffle(mask);
            assert_eq!(
                shuffled.to_array(),
                [
                    160_u8 as i8,
                    150_u8 as i8,
                    140_u8 as i8,
                    130_u8 as i8,
                    120,
                    110,
                    100,
                    90,
                    80,
                    70,
                    60,
                    50,
                    40,
                    30,
                    20,
                    10
                ]
            );

            let mask = i8x16::from_array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            let shuffled = v.shuffle(mask);
            assert_eq!(shuffled.to_array(), [10; 16]);
        }
    }

    #[test]
    fn i16x8_get() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        unsafe {
            let v = i16x8::from_array([10, 20, 30, 40, 50, 60, 70, 80]);
            assert_eq!(v.get::<0>(), 10);
            assert_eq!(v.get::<7>(), 80);
        }
    }

    #[test]
    fn i16x8_set() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        unsafe {
            let v = i16x8::from_array([10, 20, 30, 40, 50, 60, 70, 80]);
            let v = v.set::<3>(99);
            assert_eq!(v.to_array(), [10, 20, 30, 99, 50, 60, 70, 80]);
        }
    }

    #[test]
    fn i16x8_clamp_to_i8_range_and_pack() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        unsafe {
            let a = i16x8::from_array([100, 200, -100, -200, 50, 300, -50, -300]);
            let b = i16x8::from_array([0, 127, -128, 0, 1, 2, 3, 4]);
            let packed = a.clamp_to_i8_range_and_pack(b);
            assert_eq!(
                packed.to_array(),
                [100, 127, -100, -128, 50, 127, -50, -128, 0, 127, -128, 0, 1, 2, 3, 4]
            );
        }
    }

    #[test]
    fn i16x8_clamp_to_u8_range_and_pack() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        unsafe {
            let a = i16x8::from_array([100, 200, -100, -200, 50, 300, -50, -300]);
            let b = i16x8::from_array([0, 127, -128, 0, 1, 2, 3, 4]);
            let packed = a.clamp_to_u8_range_and_pack(b);
            assert_eq!(
                packed.to_array(),
                [100, 200_u8 as i8, 0, 0, 50, 255_u8 as i8, 0, 0, 0, 127, 0, 0, 1, 2, 3, 4]
            );
        }
    }

    #[test]
    fn i32x4_clamp_to_i16_range_and_pack() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        unsafe {
            let a = i32x4::from_array([100, 100000, -100, -100000]);
            let b = i32x4::from_array([0, 32767, -32768, 0]);
            let packed = a.clamp_to_i16_range_and_pack(b);
            assert_eq!(packed.to_array(), [100, 32767, -100, -32768, 0, 32767, -32768, 0]);
        }
    }

    #[test]
    fn i32x4_clamp_to_u16_range_and_pack() {
        if !std::arch::is_x86_feature_detected!("sse4.1") {
            return;
        }
        unsafe {
            let a = i32x4::from_array([100, 100000, -100, -100000]);
            let b = i32x4::from_array([0, 32767, -32768, 0]);
            let packed = a.clamp_to_u16_range_and_pack(b);
            assert_eq!(packed.to_array(), [100, 65535_u16 as i16, 0, 0, 0, 32767, 0, 0]);
        }
    }

    #[test]
    fn i8x16_min_max() {
        if !std::arch::is_x86_feature_detected!("sse4.1") {
            return;
        }
        unsafe {
            let a = i8x16::from_array([1, -1, 100, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            let b = i8x16::from_array([2, -2, 50, -50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(a.min_signed(b).to_array(), [1, -2, 50, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(a.max_signed(b).to_array(), [2, -1, 100, -50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(a.min_unsigned(b).to_array(), [1, -2, 50, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(a.max_unsigned(b).to_array(), [2, -1, 100, -50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        }
    }

    #[test]
    fn i16x8_min_max() {
        if !std::arch::is_x86_feature_detected!("sse4.1") {
            return;
        }
        unsafe {
            let a = i16x8::from_array([1, -1, 1000, -1000, 0, 0, 0, 0]);
            let b = i16x8::from_array([2, -2, 500, -500, 0, 0, 0, 0]);
            assert_eq!(a.min_signed(b).to_array(), [1, -2, 500, -1000, 0, 0, 0, 0]);
            assert_eq!(a.max_signed(b).to_array(), [2, -1, 1000, -500, 0, 0, 0, 0]);
            assert_eq!(a.min_unsigned(b).to_array(), [1, -2, 500, -1000, 0, 0, 0, 0]);
            assert_eq!(a.max_unsigned(b).to_array(), [2, -1, 1000, -500, 0, 0, 0, 0]);
        }
    }

    #[test]
    fn i32x4_min_max() {
        if !std::arch::is_x86_feature_detected!("sse4.1") {
            return;
        }
        unsafe {
            let a = i32x4::from_array([1, -1, 100000, -100000]);
            let b = i32x4::from_array([2, -2, 50000, -50000]);
            assert_eq!(a.min_signed(b).to_array(), [1, -2, 50000, -100000]);
            assert_eq!(a.max_signed(b).to_array(), [2, -1, 100000, -50000]);
            assert_eq!(a.min_unsigned(b).to_array(), [1, -2, 50000, -100000]);
            assert_eq!(a.max_unsigned(b).to_array(), [2, -1, 100000, -50000]);
        }
    }

    #[test]
    fn i32x4_as_slice() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        unsafe {
            let mut a = test_array_i32x4();
            let mut b = i32x4::from_array(a);
            assert_eq!(*b.as_slice(), a);
            a[2] = 99;
            b.as_slice_mut()[2] = 99;
            assert_eq!(*b.as_slice(), a);
        }
    }

    #[test]
    fn i64x2_as_slice() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        unsafe {
            let mut a = test_array_i64x2();
            let mut b = i64x2::from_array(a);
            assert_eq!(*b.as_slice(), a);
            a[1] = 99;
            b.as_slice_mut()[1] = 99;
            assert_eq!(*b.as_slice(), a);
        }
    }

    #[test]
    fn i8x16_simd_gt_lt() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        unsafe {
            let a = i8x16::from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
            let b = i8x16::from_array([2, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16]);
            let gt = a.simd_gt(b);
            let lt = a.simd_lt(b);
            assert_eq!(
                gt.to_array().map(|x| x != 0),
                [
                    false, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false
                ]
            );
            assert_eq!(
                lt.to_array().map(|x| x != 0),
                [
                    true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false
                ]
            );
        }
    }

    #[test]
    fn i16x8_simd_gt_lt() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        unsafe {
            let a = i16x8::from_array([1, 2, 3, 4, 5, 6, 7, 8]);
            let b = i16x8::from_array([2, 2, 2, 4, 4, 6, 6, 8]);
            let gt = a.simd_gt(b);
            let lt = a.simd_lt(b);
            assert_eq!(gt.to_array().map(|x| x != 0), [false, false, true, false, true, false, true, false]);
            assert_eq!(
                lt.to_array().map(|x| x != 0),
                [true, false, false, false, false, false, false, false]
            );
        }
    }

    #[test]
    fn i32x4_simd_gt_lt() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        unsafe {
            let a = i32x4::from_array([1, 2, 3, 4]);
            let b = i32x4::from_array([2, 2, 2, 4]);
            let gt = a.simd_gt(b);
            let lt = a.simd_lt(b);
            assert_eq!(gt.to_array().map(|x| x != 0), [false, false, true, false]);
            assert_eq!(lt.to_array().map(|x| x != 0), [true, false, false, false]);
        }
    }

    #[test]
    fn i64x2_simd_gt_lt() {
        if !std::arch::is_x86_feature_detected!("sse4.2") {
            return;
        }
        unsafe {
            let a = i64x2::from_array([1, 2]);
            let b = i64x2::from_array([2, 2]);
            let gt = a.simd_gt(b);
            let lt = a.simd_lt(b);
            assert_eq!(gt.to_array().map(|x| x != 0), [false, false]);
            assert_eq!(lt.to_array().map(|x| x != 0), [true, false]);
        }
    }
}
