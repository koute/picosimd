#![allow(clippy::should_implement_trait)]

macro_rules! zip_map {
    ($lhs:expr, $rhs:expr, $len:expr, |$a:ident, $b:ident| $body:expr) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        let mut i = 0;
        let mut result = [0; $len];
        while i < $len {
            let $a = lhs[i];
            let $b = rhs[i];
            result[i] = $body;
            i += 1;
        }
        result
    }};
}

macro_rules! array_map {
    ($array:expr, $len:expr, |$item:ident| $body:expr) => {{
        let array = $array;
        let mut i = 0;
        let mut result = [0; $len];
        while i < $len {
            let $item = array[i];
            result[i] = $body;
            i += 1;
        }
        result
    }};
}

macro_rules! array_map_indexed {
    ($len:expr, |$index:ident| $body:expr) => {{
        let mut i = 0;
        let mut result = [0; $len];
        while i < $len {
            let $index = i;
            result[i] = $body;
            i += 1;
        }
        result
    }};
}

macro_rules! array_map3 {
    ($a:expr, $b:expr, $c:expr, $len:expr, |$ai:ident, $bi:ident, $ci:ident| $body:expr) => {{
        let a = $a;
        let b = $b;
        let c = $c;
        let mut i = 0;
        let mut result = [0; $len];
        while i < $len {
            let $ai = a[i];
            let $bi = b[i];
            let $ci = c[i];
            result[i] = $body;
            i += 1;
        }
        result
    }};
}

macro_rules! reduce {
    ($array:expr, $len:expr, $ty:ty, $init:expr, |$acc:ident, $item:ident| $body:expr) => {{
        let array = $array;
        let mut i = 0;
        let mut $acc: $ty = $init;
        while i < $len {
            let $item = array[i];
            $acc = $body;
            i += 1;
        }
        $acc
    }};
}

macro_rules! reduce_indexed {
    ($array:expr, $len:expr, $ty:ty, $init:expr, |$acc:ident, $item:ident, $index:ident| $body:expr) => {{
        let array = $array;
        let mut i = 0;
        let mut $acc: $ty = $init;
        while i < $len {
            let $item = array[i];
            let $index = i;
            $acc = $body;
            i += 1;
        }
        $acc
    }};
}

macro_rules! impl_fallback {
    (
        $type:ident,
        $lane_ty:ty,
        $lane_ty_unsigned:ty,
        $lane_count:expr
    ) => {
        #[repr(transparent)]
        #[derive(Copy, Clone)]
        pub struct $type(pub [$lane_ty; $lane_count]);

        impl $type {
            #[inline]
            pub const fn zero() -> Self {
                Self([0; $lane_count])
            }

            #[inline]
            pub const fn negative_one() -> Self {
                Self([-1; $lane_count])
            }

            #[inline]
            pub const fn splat(value: $lane_ty) -> Self {
                Self([value; $lane_count])
            }

            #[inline]
            pub const fn from_array(xs: [$lane_ty; $lane_count]) -> Self {
                Self(xs)
            }

            #[inline]
            pub const fn from_array_ref(xs: &[$lane_ty; $lane_count]) -> Self {
                Self(*xs)
            }

            #[inline]
            pub const fn to_array(self) -> [$lane_ty; $lane_count] {
                self.0
            }

            #[inline]
            pub const fn as_slice(&self) -> &[$lane_ty; $lane_count] {
                &self.0
            }

            #[inline]
            pub const fn as_slice_mut(&mut self) -> &mut [$lane_ty; $lane_count] {
                &mut self.0
            }

            #[inline]
            pub const fn from_fallback(value: $type) -> Self {
                value
            }

            #[inline]
            pub const fn is_equal(self, rhs: Self) -> bool {
                let mut i = 0;
                while i < $lane_count {
                    if self.0[i] != rhs.0[i] {
                        return false;
                    }
                    i += 1;
                }
                true
            }

            #[inline]
            pub const fn is_zero(self) -> bool {
                self.is_equal(Self::zero())
            }

            #[inline]
            pub const fn simd_eq(self, rhs: Self) -> Self {
                Self(zip_map!(self.0, rhs.0, $lane_count, |lhs, rhs| if lhs == rhs {
                    -1
                } else {
                    0
                }))
            }

            #[inline]
            pub const fn simd_gt(self, rhs: Self) -> Self {
                Self(zip_map!(self.0, rhs.0, $lane_count, |lhs, rhs| if lhs > rhs {
                    -1
                } else {
                    0
                }))
            }

            #[inline]
            pub const fn simd_lt(self, rhs: Self) -> Self {
                rhs.simd_gt(self)
            }

            #[inline]
            pub const fn and_not(self, rhs: Self) -> Self {
                Self(zip_map!(self.0, rhs.0, $lane_count, |lhs, rhs| lhs & (!rhs)))
            }

            #[inline]
            pub const fn and(self, rhs: Self) -> Self {
                Self(zip_map!(self.0, rhs.0, $lane_count, |lhs, rhs| lhs & rhs))
            }

            #[inline]
            pub const fn or(self, rhs: Self) -> Self {
                Self(zip_map!(self.0, rhs.0, $lane_count, |lhs, rhs| lhs | rhs))
            }

            #[inline]
            pub const fn xor(self, rhs: Self) -> Self {
                Self(zip_map!(self.0, rhs.0, $lane_count, |lhs, rhs| lhs ^ rhs))
            }

            #[inline]
            pub const fn add(self, rhs: Self) -> Self {
                Self(zip_map!(self.0, rhs.0, $lane_count, |lhs, rhs| lhs.wrapping_add(rhs)))
            }

            #[inline]
            pub const fn sub(self, rhs: Self) -> Self {
                Self(zip_map!(self.0, rhs.0, $lane_count, |lhs, rhs| lhs.wrapping_sub(rhs)))
            }

            #[inline]
            pub const fn saturating_add(self, rhs: Self) -> Self {
                Self(zip_map!(self.0, rhs.0, $lane_count, |lhs, rhs| lhs.saturating_add(rhs)))
            }

            #[inline]
            pub const fn saturating_add_unsigned(self, rhs: Self) -> Self {
                Self(zip_map!(self.0, rhs.0, $lane_count, |lhs, rhs| {
                    (lhs as $lane_ty_unsigned).saturating_add(rhs as $lane_ty_unsigned) as $lane_ty
                }))
            }

            #[inline]
            pub const fn saturating_sub(self, rhs: Self) -> Self {
                Self(zip_map!(self.0, rhs.0, $lane_count, |lhs, rhs| lhs.saturating_sub(rhs)))
            }

            #[inline]
            pub const fn saturating_sub_unsigned(self, rhs: Self) -> Self {
                Self(zip_map!(self.0, rhs.0, $lane_count, |lhs, rhs| {
                    (lhs as $lane_ty_unsigned).saturating_sub(rhs as $lane_ty_unsigned) as $lane_ty
                }))
            }

            #[inline]
            pub const fn wrapping_reduce(self) -> $lane_ty {
                reduce!(self.0, $lane_count, $lane_ty, 0, |acc, item| acc.wrapping_add(item))
            }

            #[inline]
            pub const fn bitwise_reduce(self) -> $lane_ty {
                reduce!(self.0, $lane_count, $lane_ty, 0, |acc, item| acc | item)
            }

            #[inline]
            pub const fn horizontal_max_signed(self) -> $lane_ty {
                reduce!(self.0, $lane_count, $lane_ty, self.0[0], |acc, item| if item > acc {
                    item
                } else {
                    acc
                })
            }

            #[inline]
            pub const fn horizontal_min_signed(self) -> $lane_ty {
                reduce!(self.0, $lane_count, $lane_ty, self.0[0], |acc, item| if item < acc {
                    item
                } else {
                    acc
                })
            }

            #[inline]
            pub const fn horizontal_max_unsigned(self) -> $lane_ty_unsigned {
                reduce!(
                    self.0,
                    $lane_count,
                    $lane_ty_unsigned,
                    self.0[0] as $lane_ty_unsigned,
                    |acc, item| {
                        let item = item as $lane_ty_unsigned;
                        if item > acc { item } else { acc }
                    }
                )
            }

            #[inline]
            pub const fn horizontal_min_unsigned(self) -> $lane_ty_unsigned {
                reduce!(
                    self.0,
                    $lane_count,
                    $lane_ty_unsigned,
                    self.0[0] as $lane_ty_unsigned,
                    |acc, item| {
                        let item = item as $lane_ty_unsigned;
                        if item < acc { item } else { acc }
                    }
                )
            }

            #[must_use]
            #[inline]
            pub const fn set_dynamic(self, index: $lane_ty_unsigned, value: $lane_ty) -> Self {
                let mut copy = self;
                if index < $lane_count {
                    copy.0[index as usize] = value;
                }
                copy
            }

            #[must_use]
            #[inline]
            pub const fn get_dynamic(self, index: $lane_ty_unsigned) -> $lane_ty {
                if index < $lane_count { self.0[index as usize] } else { 0 }
            }

            #[inline]
            pub const fn min_unsigned(self, rhs: Self) -> Self {
                Self(zip_map!(self.0, rhs.0, $lane_count, |lhs, rhs| {
                    let lhs_unsigned = lhs as $lane_ty_unsigned;
                    let rhs_unsigned = rhs as $lane_ty_unsigned;
                    if lhs_unsigned < rhs_unsigned { lhs } else { rhs }
                }))
            }

            #[inline]
            pub const fn max_unsigned(self, rhs: Self) -> Self {
                Self(zip_map!(self.0, rhs.0, $lane_count, |lhs, rhs| {
                    let lhs_unsigned = lhs as $lane_ty_unsigned;
                    let rhs_unsigned = rhs as $lane_ty_unsigned;
                    if lhs_unsigned > rhs_unsigned { lhs } else { rhs }
                }))
            }

            #[inline]
            pub const fn min_signed(self, rhs: Self) -> Self {
                Self(zip_map!(self.0, rhs.0, $lane_count, |lhs, rhs| if lhs < rhs {
                    lhs
                } else {
                    rhs
                }))
            }

            #[inline]
            pub const fn max_signed(self, rhs: Self) -> Self {
                Self(zip_map!(self.0, rhs.0, $lane_count, |lhs, rhs| if lhs > rhs {
                    lhs
                } else {
                    rhs
                }))
            }

            #[inline]
            pub const unsafe fn load_unaligned(address: *const u8) -> Self {
                unsafe { core::ptr::read_unaligned(address.cast()) }
            }
        }

        #[cfg(feature = "ops")]
        impl core::fmt::Debug for $type {
            fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
                self.to_array().fmt(fmt)
            }
        }

        #[cfg(feature = "ops")]
        impl core::ops::Add for $type {
            type Output = Self;

            #[inline]
            fn add(self, rhs: Self) -> Self {
                Self::add(self, rhs)
            }
        }

        #[cfg(feature = "ops")]
        impl core::ops::Sub for $type {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self {
                Self::sub(self, rhs)
            }
        }

        #[cfg(feature = "ops")]
        impl core::ops::AddAssign for $type {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        #[cfg(feature = "ops")]
        impl core::ops::SubAssign for $type {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        #[cfg(feature = "ops")]
        impl PartialEq for $type {
            fn eq(&self, rhs: &Self) -> bool {
                self.is_equal(*rhs)
            }
        }

        #[cfg(feature = "ops")]
        impl Eq for $type {}

        #[cfg(feature = "ops")]
        impl core::ops::BitAnd for $type {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                self.and(rhs)
            }
        }

        #[cfg(feature = "ops")]
        impl core::ops::BitOr for $type {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                self.or(rhs)
            }
        }

        #[cfg(feature = "ops")]
        impl core::ops::BitXor for $type {
            type Output = Self;

            #[inline]
            fn bitxor(self, rhs: Self) -> Self {
                self.xor(rhs)
            }
        }
    };
}

impl i8x32 {
    #[inline]
    pub const fn from_i1x32_sext(packed_mask: i32) -> Self {
        Self(array_map_indexed!(32, |n| if (packed_mask & (1 << n)) != 0 { -1 } else { 0 }))
    }

    #[inline]
    pub const fn to_i16x32_sext(self) -> i16x32 {
        i16x32(array_map!(self.0, 32, |item| item as i16))
    }

    #[inline]
    pub const fn to_i32x32_sext(self) -> i32x32 {
        i32x32(array_map!(self.0, 32, |item| item as i32))
    }

    #[inline]
    pub const fn most_significant_bits(self) -> i32 {
        reduce_indexed!(self.0, 32, i32, 0, |acc, item, index| acc | (((item as u8) >> 7) as i32) << index)
    }

    #[must_use]
    #[inline]
    pub const fn conditional_assign(self, rhs: Self, should_pick_rhs: Self) -> Self {
        Self(array_map3!(self.0, rhs.0, should_pick_rhs.0, 32, |a, b, c| if c != 0 {
            b
        } else {
            a
        }))
    }
}

impl i8x16 {
    #[inline]
    pub const fn from_i1x16_sext(packed_mask: i16) -> Self {
        Self(array_map_indexed!(16, |n| if (packed_mask & (1 << n)) != 0 { -1 } else { 0 }))
    }

    #[inline]
    pub const fn to_i16x16_sext(self) -> i16x16 {
        i16x16(array_map!(self.0, 16, |item| item as i16))
    }
}

impl i16x32 {
    #[inline]
    pub const fn to_i32x32_sext(self) -> i32x32 {
        i32x32(array_map!(self.0, 32, |item| item as i32))
    }
}

impl i32x8 {
    #[inline]
    pub const fn from_i1x8_sext(packed_mask: i8) -> Self {
        Self(array_map_indexed!(8, |n| if (packed_mask & (1 << n)) != 0 { -1 } else { 0 }))
    }
}

impl i32x32 {
    #[inline]
    pub const fn from_i8x32_sext(value: i8x32) -> Self {
        Self(array_map!(value.0, 32, |item| item as i32))
    }
}

impl_fallback!(i8x64, i8, u8, 64);
impl_fallback!(i8x32, i8, u8, 32);
impl_fallback!(i8x16, i8, u8, 16);
impl_fallback!(i16x32, i16, u16, 32);
impl_fallback!(i16x16, i16, u16, 16);
impl_fallback!(i16x8, i16, u16, 8);
impl_fallback!(i32x32, i32, u32, 32);
impl_fallback!(i32x16, i32, u32, 16);
impl_fallback!(i32x8, i32, u32, 8);
impl_fallback!(i32x4, i32, u32, 4);
impl_fallback!(i64x8, i64, u64, 8);
impl_fallback!(i64x4, i64, u64, 4);
impl_fallback!(i64x2, i64, u64, 2);

macro_rules! impl_clamp_to_i8_range {
    ($type:ty, $target_ty:ident, $lane_count:expr) => {
        impl $type {
            #[inline]
            pub const fn clamp_to_i8_range(self) -> $target_ty {
                $target_ty(array_map!(self.0, $lane_count, |value| if value > 127 {
                    127
                } else if value < -128 {
                    -128
                } else {
                    value as i8
                }))
            }
        }
    };
}

impl_clamp_to_i8_range!(i16x16, i8x16, 16);
impl_clamp_to_i8_range!(i16x32, i8x32, 32);
impl_clamp_to_i8_range!(i32x32, i8x32, 32);
