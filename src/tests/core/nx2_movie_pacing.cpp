// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "core/hle/service/nx2_movie_pacing.h"
#include "core/hle/service/nx2_present_pacer.h"

TEST_CASE("MKW movie pacing is limited to recent large frames in the current session", "[nx2]") {
    using Service::MkwMovieActivity;
    using namespace std::chrono_literals;
    CHECK(MkwMovieActivity::IsLargeMovie(0x0400C3F00006E000ULL, 2560));
    CHECK(MkwMovieActivity::IsLargeMovie(0x0400C3F00006E000ULL, 1920));
    CHECK_FALSE(MkwMovieActivity::IsLargeMovie(0x0400C3F00006E000ULL, 1919));
    CHECK_FALSE(MkwMovieActivity::IsLargeMovie(0x0400C3F00006E000ULL, 640));
    CHECK_FALSE(MkwMovieActivity::IsLargeMovie(0x0400123400000000ULL, 2560));

    MkwMovieActivity activity;
    MkwMovieActivity other_session;
    const MkwMovieActivity::Clock::time_point start{1s};
    CHECK_FALSE(activity.Active(start));
    activity.RecordFrame(start);
    CHECK(activity.Active(start));
    CHECK(activity.Active(start + 249ms));
    CHECK_FALSE(activity.Active(start + 250ms));
    CHECK_FALSE(activity.Active(start - 1ms));
    CHECK_FALSE(other_session.Active(start));
    activity.RecordFrame(start + 200ms);
    CHECK(activity.Active(start + 300ms));
    activity.Reset();
    CHECK_FALSE(activity.Active(start + 300ms));
}

TEST_CASE("NX2 presentation reanchors after an unpaced movie", "[core][nx2_present_pacer]") {
    using namespace std::chrono_literals;
    Service::Nx2PresentPacer pacer;
    const Service::Nx2PresentPacer::Clock::time_point start{1s};
    pacer.Advance(start, 33ms);
    REQUIRE(pacer.Advance(start + 1ms, 33ms).deadline == start + 33ms);
    pacer = {};
    const auto resumed = pacer.Advance(start + 2ms, 16ms);
    CHECK(resumed.reset);
    CHECK(resumed.deadline == start + 2ms);
    CHECK(pacer.Advance(start + 3ms, 16ms).deadline == start + 18ms);
}
