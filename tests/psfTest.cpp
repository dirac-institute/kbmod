/*
 * SpreadFunc_test.cpp
 *
 *  Created on: Jul 11, 2017
 *      Author: kbmod-usr
 */

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "PointSpreadFunc.h"

namespace kbmod {

TEST_CASE("Point Spread makes gaussians", "[PSF]" )
{

	PointSpreadFunc a(1.0);
	PointSpreadFunc b(1.0);
	PointSpreadFunc c(1.5);

	REQUIRE( a.getDim() == b.getDim() );

	//float aCenter =

}

} /* namespace kbmod */

