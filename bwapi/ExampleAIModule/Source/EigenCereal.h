#pragma once
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <Eigen/Dense>

namespace cereal
{
	template <class Archive, class Derived> inline
		typename std::enable_if<traits::is_output_serializable<BinaryData<typename Derived::Scalar>, Archive>::value, void>::type
		save(Archive & ar, Eigen::PlainObjectBase<Derived> const & m) {
		typedef Eigen::PlainObjectBase<Derived> ArrT;
		if (ArrT::RowsAtCompileTime == Eigen::Dynamic) ar(m.rows());
		if (ArrT::ColsAtCompileTime == Eigen::Dynamic) ar(m.cols());
		ar(binary_data(m.data(), m.size() * sizeof(typename Derived::Scalar)));
	}

	template <class Archive, class Derived> inline
		typename std::enable_if<traits::is_input_serializable<BinaryData<typename Derived::Scalar>, Archive>::value, void>::type
		load(Archive & ar, Eigen::PlainObjectBase<Derived> & m) {
		typedef Eigen::PlainObjectBase<Derived> ArrT;
		Eigen::Index rows = ArrT::RowsAtCompileTime, cols = ArrT::ColsAtCompileTime;
		if (rows == Eigen::Dynamic) ar(rows);
		if (cols == Eigen::Dynamic) ar(cols);
		m.resize(rows, cols);
		ar(binary_data(m.data(), static_cast<std::size_t>(rows*cols * sizeof(typename Derived::Scalar))));
	}
}
